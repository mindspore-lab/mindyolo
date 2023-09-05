import math
import os
import sys
import time
from typing import Union, Tuple, List

import numpy as np
from mindspore import Profiler, SummaryRecord, Tensor
from mindyolo.utils.modelarts import sync_data
from mindyolo.utils import CheckpointManager, logger
from mindyolo.utils.registry import Registry
from mindyolo.utils.train_step_factory import create_train_step_fn

CALLBACK_REGISTRY = Registry("callback")


def create_callback(arg_callback):
    def _create_callback_worker(name, **kwargs):
        cb_cls = CALLBACK_REGISTRY.get(name)
        instance = cb_cls(**kwargs)
        return instance

    assert isinstance(arg_callback, (tuple, list)), f'expect callback to be list of tuple, ' \
                                                     f'but got {type(arg_callback)} instead'
    for i, cb in enumerate(arg_callback):
        assert isinstance(cb, dict) and 'name' in cb, f'callback[{i}] is not a dict or does not contain key [name]'

    logger.info(CALLBACK_REGISTRY)

    return [_create_callback_worker(**kw) for kw in arg_callback]


class RunContext:
    """
    Hold and manage information about the running state of the model
    Args:
        epoch_num (int): total epoch number in the training process
        steps_per_epoch (int): total steps of one epoch
        trainer (Trainer): trainer class that perform training process
        test_fn (Function): test function that can evaluate the training model
        enable_modelarts (bool): whether to enable modelarts. usually on cloud when true
        ckpt_save_dir (str): checkpoint saving directory
        train_url (str): training url. usually on cloud when not empty

    """

    def __init__(
        self,
        epoch_num=0,
        steps_per_epoch=0,
        total_steps=0,
        trainer=None,
        test_fn=None,
        enable_modelarts=False,
        ckpt_save_dir="",
        save_dir="",
        train_url="",
        overflow_still_update=False,
        ms_jit=True,
        rank_size=8,
    ):

        self.epoch_num = epoch_num
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_steps
        self.trainer = trainer
        self.test_fn = test_fn
        self.ckpt_save_dir = ckpt_save_dir
        self.save_dir = save_dir
        self.enable_modelarts = enable_modelarts
        self.train_url = train_url
        self.overflow_still_update = overflow_still_update
        self.ms_jit = ms_jit
        self.rank_size = rank_size

        # the first index start with 1 rather than 0
        self.cur_epoch_index = 0
        self.cur_step_index = 0
        self.loss = []
        self.lr = 0


class BaseCallback:
    """
    Base class of callback. Applied in Train function, it can take actions on 6 different stage of the training process.

    """

    def __init__(self):
        pass

    def __repr__(self):
        members = vars(self)
        mem_str = ", ".join([f"{k}={v}" for k, v in members.items()])
        fmt_str = self.__class__.__name__ + f"({mem_str})"
        return fmt_str

    def on_train_begin(self, run_context: RunContext):
        """hooks to run on the beginning of training process"""
        pass

    def on_train_end(self, run_context: RunContext):
        """hooks to run on the end of training process"""
        pass

    def on_train_epoch_begin(self, run_context: RunContext):
        """hooks to run on the beginning of a training epoch"""
        pass

    def on_train_epoch_end(self, run_context: RunContext):
        """hooks to run on the end of a training epoch"""
        pass

    def on_train_step_begin(self, run_context: RunContext):
        """hooks to run on the beginning of a training step"""
        pass

    def on_train_step_end(self, run_context: RunContext):
        """hooks to run on the end of a training step"""
        pass


@CALLBACK_REGISTRY.registry_module()
class YoloxSwitchTrain(BaseCallback):
    """
    Switch train hook applied in yolox model. Yolox model uses a two stage training strategy. Compared with the 1st
    stage, the 2nd second has no mosaic data augmentation and add l1 loss item. Reference: url

    Args:
        switch_epoch_num (int): index of epoch to switch stage. This value equals to the epoch number of first stage.
        is_switch_loss (bool): whether to switch loss
        is_switch_data_aug (bool): whether to switch data augmentation

    """

    def __init__(self, switch_epoch_num=285, is_switch_loss=True, is_switch_data_aug=False, **kwargs):
        super().__init__()
        self.switch_epoch_num = switch_epoch_num
        self.switch_epoch_index = switch_epoch_num + 1
        self.is_switch_loss = is_switch_loss
        self.is_switch_data_aug = is_switch_data_aug

    def on_train_step_begin(self, run_context: RunContext):
        pass

    def on_train_epoch_begin(self, run_context: RunContext):
        cur_epoch_index = run_context.cur_epoch_index
        trainer = run_context.trainer
        loss_ratio = run_context.rank_size
        overflow_still_update = run_context.overflow_still_update
        ms_jit = run_context.ms_jit

        # switch loss
        if self.is_switch_loss and cur_epoch_index == self.switch_epoch_index:
            logger.info(f"\nAdding L1 loss starts from epoch {self.switch_epoch_index}. Graph recompiling\n")
            trainer.loss_fn.use_l1 = True
            trainer.train_step_fn = create_train_step_fn(task='detect',
                                                         network=trainer.network,
                                                         loss_fn=trainer.loss_fn,
                                                         optimizer=trainer.optimizer,
                                                         loss_ratio=loss_ratio,
                                                         scaler=trainer.scaler,
                                                         reducer=trainer.reducer,
                                                         ema=trainer.ema,
                                                         overflow_still_update=overflow_still_update,
                                                         ms_jit=ms_jit)

        # switch data_aug, not implemented hear
        if self.is_switch_data_aug:
            raise ValueError(
                "Currently switch_data_aug should be implemented using multi-stage training pipe line. "
                "Refer train_transforms for more information. Keep is_switch_data_aug button False."
            )


@CALLBACK_REGISTRY.registry_module()
class EvalWhileTrain(BaseCallback):
    """
    Callback of evaluation while training. Mainly two parts are included, namely evaluating at requested time and
    uploading ckpt file to cloud. Piecewise evaluation with different interval in each piece is supported.
    Args:
        stage_epochs (Union(List, Tuple, int)): For list or tuple type, piecewise mode is on and each element
                indicates the epoch number in its piece. For int type, single piece mode is on and the value indicates
                the possible max epoch index where the model will be evaluated. Default positive infinite means no switch
        stage_intervals (Union(List, Tuple, int)): With the same type and length with stage_epochs, interval represents
                the corresponding interval of each piece. Default 1
        eval_last_epoch (bool): whether to evaluate the last epoch of each piece. Default True
        isolated_epochs (Union(List, Tuple, int, None)): isolated epochs to evaluation for flexible sense. Default None.
        keep_checkpoint_max (int): the most possible checkpoint to keep on disk. Default 10.

    Example:
        Case 1: evaluate single stage
        >>> hook EvalWhileTrain(stage_intervals=5)
        The above hook will evaluate the model with an interval of 5, and final epoch will be evaluated by default.

        Case 2: evaluate multiple stage
        >>> hook = EvalWhileTrain(stage_epochs=[285, 15], stage_intervals=[25, 5], isolated_epochs=[3, 213])
        The above hook will evaluate the model by two stage. At 1st stage, 285 epochs are evaluated with an interval of
            25, while at 2nd stage, 15 epochs are evaluated with an interval of 5. Meanwhile, the model is evaluated at
            3 and 213 epoch specified by isolated_epochs. The final epoch of the two stage, namely 285 and 300, will be
            evaluated by default.
    """

    def __init__(
        self,
        stage_epochs: Union[List, Tuple, int] = sys.maxsize,
        stage_intervals: Union[List, Tuple, int] = 1,
        eval_last_epoch=True,
        isolated_epochs: Union[List, Tuple, int, None] = None,
        keep_checkpoint_max=10,
    ):
        super().__init__()
        assert isinstance(stage_intervals, (list, tuple, int))
        assert isinstance(stage_epochs, (list, tuple, int))

        # cast interval list in case of 1 stage
        if isinstance(stage_intervals, int) or isinstance(stage_epochs, int):
            assert isinstance(stage_intervals, int) and isinstance(
                stage_epochs, int
            ), f"stage_intervals and stage_epochs must be int at the same time"
            stage_intervals = [stage_intervals]
            stage_epochs = [stage_epochs]

        # cast isolated_epochs to list
        if isolated_epochs is not None:
            assert isinstance(isolated_epochs, (list, tuple, int))
            if isinstance(isolated_epochs, int):
                isolated_epochs = [isolated_epochs]
        else:
            isolated_epochs = []

        assert len(stage_intervals) == len(stage_epochs)
        self.stage_intervals = stage_intervals
        self.stage_epochs = stage_epochs  # for log
        self.stage_cum_epochs = np.cumsum(stage_epochs)
        self.eval_last_epoch = eval_last_epoch
        self.isolated_epochs = isolated_epochs
        self.keep_checkpoint_max = keep_checkpoint_max
        self.manager_best = CheckpointManager(ckpt_save_policy="top_k")
        self.ckpt_filelist_best = []

    def on_train_epoch_end(self, run_context: RunContext):
        cur_epoch_index = run_context.cur_epoch_index
        epochs = run_context.epoch_num
        # reset to total epoch if exceed
        for i in range(len(self.stage_cum_epochs)):
            if self.stage_cum_epochs[i] > epochs:
                self.stage_cum_epochs[i] = epochs

        stage = np.searchsorted(self.stage_cum_epochs, cur_epoch_index, side="left")
        # in case of cur_epoch_index greater than total epoch that need evaluation
        if stage == len(self.stage_intervals):
            return

        offset = self.stage_cum_epochs[stage - 1] if stage > 0 else 0
        interval_cond = (cur_epoch_index - offset) % self.stage_intervals[stage] == 0
        last_cond = self.eval_last_epoch and (cur_epoch_index == self.stage_cum_epochs[stage])
        isolated_cond = any(cur_epoch_index == e for e in self.isolated_epochs)
        if interval_cond or last_cond or isolated_cond:
            self._run_eval(run_context)

    def on_train_end(self, run_context: RunContext):
        enable_modelarts = run_context.enable_modelarts
        train_url = run_context.train_url
        if enable_modelarts and self.ckpt_filelist_best:
            ckpt_filelist_best = [s[0] for s in self.ckpt_filelist_best]
            for p in ckpt_filelist_best:
                sync_data(p, train_url + "/weights/" + p.split("/")[-1])

    def _run_eval(self, run_context: RunContext):
        s_eval_time = time.time()

        trainer = run_context.trainer
        test_fn = run_context.test_fn
        cur_epoch = run_context.cur_epoch_index
        epochs = run_context.epoch_num
        ckpt_save_dir = run_context.ckpt_save_dir

        eval_network = trainer.ema.ema if trainer.ema else trainer.network
        _train_status = eval_network.training
        eval_network.set_train(False)
        accuracy = test_fn(network=eval_network, cur_epoch=f'{cur_epoch:03d}')
        accuracy = accuracy[0] if isinstance(accuracy, (list, tuple)) else accuracy
        eval_network.set_train(_train_status)

        save_path_best = os.path.join(
            ckpt_save_dir,
            f"best_{trainer.model_name}-{cur_epoch}_{trainer.steps_per_epoch}" f"_acc{accuracy:.3f}.ckpt",
        )

        if trainer.main_device:
            self.ckpt_filelist_best = self.manager_best.save_ckpoint(
                eval_network, num_ckpt=self.keep_checkpoint_max, metric=accuracy, save_path=save_path_best
            )
            best_path, best_accu = self.ckpt_filelist_best[0]
            logger.info(
                f"Epoch {cur_epoch}/{epochs}, eval accuracy: {accuracy:.3f}, "
                f"run_eval time: {(time.time() - s_eval_time):.3f} s."
            )
            logger.info(f"best accuracy: {best_accu:.3f}, saved at: {best_path}")


@CALLBACK_REGISTRY.registry_module()
class SummaryCallback(BaseCallback):
    """
    Callback of whether to collect summary data at training time.
    """

    def __init__(self):
        super().__init__()

    def on_train_begin(self, run_context: RunContext):
        """hooks to run on the beginning of training process"""
        self.summary_dir = os.path.join(run_context.save_dir, "summary")
        self.summary_record = SummaryRecord(self.summary_dir)

    def on_train_end(self, run_context: RunContext):
        """hooks to run on the end of training process"""
        self.summary_record.close()
        if run_context.enable_modelarts:
            for p in os.listdir(self.summary_dir):
                summary_file_path = os.path.join(self.summary_dir, p)
                sync_data(summary_file_path, run_context.train_url + "/summary/" + summary_file_path.split("/")[-1])

    def on_train_epoch_end(self, run_context: RunContext):
        """hooks to run on the end of a training epoch"""
        trainer = run_context.trainer
        if trainer.data_sink:
            for i in range(len(run_context.loss)):
                self.summary_record.add_value("scalar", f"{trainer.loss_item_name[i]}", run_context.loss[i])
            self.summary_record.add_value("scalar", f"cur_lr", Tensor(run_context.lr))
            self.summary_record.record(run_context.cur_epoch_index)
            self.summary_record.flush()

    def on_train_step_end(self, run_context: RunContext):
        """hooks to run on the end of a training step"""
        trainer = run_context.trainer
        if run_context.cur_step_index % trainer.log_interval == 0:
            for i in range(len(run_context.loss)):
                self.summary_record.add_value("scalar", f"{trainer.loss_item_name[i]}", run_context.loss[i])
            self.summary_record.add_value("scalar", f"cur_lr", Tensor(run_context.lr))
            self.summary_record.record(run_context.cur_step_index)
            self.summary_record.flush()


@CALLBACK_REGISTRY.registry_module()
class ProfilerCallback(BaseCallback):
    """
    Callback of whether to collect profiler data at training time.

    Example:
        Case 1: Non-data sinking mode Collects performance data in the specified step interval.
        Case 2: Data sink mode Collects performance data for a specified epoch interval.
    """

    def __init__(self, profiler_step_num):
        super().__init__()
        self.profiler_step_num = profiler_step_num

    def on_train_begin(self, run_context: RunContext):
        """hooks to run on the beginning of training process"""
        self.prof_dir = os.path.join(run_context.save_dir, "profiling_data")
        self.prof = Profiler(output_path=self.prof_dir)

    def on_train_epoch_end(self, run_context: RunContext):
        """hooks to run on the beginning of a training epoch"""
        if run_context.cur_epoch_index == math.ceil(self.profiler_step_num/run_context.steps_per_epoch):
            self.prof.stop()
            self.prof.analyse()

    def on_train_step_end(self, run_context: RunContext):
        """hooks to run on the beginning of a training step"""
        if run_context.cur_step_index == self.profiler_step_num:
            self.prof.stop()
            self.prof.analyse()

    def on_train_end(self, run_context: RunContext):
        if run_context.enable_modelarts:
            for p in os.listdir(self.prof_dir):
                prof_file_path = os.path.join(self.prof_dir, p)
                sync_data(prof_file_path, run_context.train_url + "/profiling_data/" + prof_file_path.split("/")[-1])
