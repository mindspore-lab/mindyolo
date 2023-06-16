import math
import os
import time
import types
from typing import Union, List

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindyolo.utils import logger
from mindyolo.utils.callback import BaseCallback, EvalWhileTrain, RunContext
from mindyolo.utils.checkpoint_manager import CheckpointManager
from mindyolo.utils.modelarts import sync_data

__all__ = [
    "create_trainer",
]


def create_trainer(
    model_name: str,
    train_step_fn: types.FunctionType,
    scaler,
    network: nn.Cell,
    loss_fn: nn.Cell,
    ema: nn.Cell,
    optimizer: nn.Cell,
    dataloader: ms.dataset.Dataset,
    steps_per_epoch: int,
    callback: List[BaseCallback],
    reducer,
    data_sink,
    profiler
):
    return Trainer(
        model_name=model_name,
        train_step_fn=train_step_fn,
        scaler=scaler,
        network=network,
        loss_fn=loss_fn,
        ema=ema,
        optimizer=optimizer,
        dataloader=dataloader,
        steps_per_epoch=steps_per_epoch,
        callback=callback,
        reducer=reducer,
        data_sink=data_sink,
        profiler=profiler
    )


class Trainer:
    def __init__(
        self,
        model_name,
        train_step_fn,
        scaler,
        network,
        loss_fn,
        ema,
        optimizer,
        dataloader,
        steps_per_epoch,
        callback,
        reducer,
        data_sink,
        profiler
    ):
        self.model_name = model_name
        self.train_step_fn = train_step_fn
        self.scaler = scaler
        self.dataloader = dataloader
        self.network = network  # for save checkpoint
        self.loss_fn = loss_fn
        self.ema = ema  # for save checkpoint and ema
        self.optimizer = optimizer  # for save checkpoint
        self.global_step = 0
        self.steps_per_epoch = steps_per_epoch
        self.callback = callback
        self.reducer = reducer
        self.data_sink = data_sink
        self.profiler = profiler

    def train(
        self,
        epochs: int,
        main_device: bool,
        warmup_step: int = 0,
        warmup_momentum: Union[list, None] = None,
        accumulate: int = 1,
        overflow_still_update: bool = False,
        keep_checkpoint_max: int = 10,
        log_interval: int = 1,
        loss_item_name: list = [],
        save_dir: str = "",
        enable_modelarts: bool = False,
        train_url: str = "",
        run_eval: bool = False,
        test_fn: types.FunctionType = None,
        ms_jit: bool = True,
        rank_size: int = 8,
        profiler_step_num: int = 1
    ):
        # Attr
        self.epochs = epochs
        self.main_device = main_device
        self.log_interval = log_interval
        self.overflow_still_update = overflow_still_update
        self.loss_item_name = loss_item_name
        self.profiler_step_num = profiler_step_num

        # Directories
        ckpt_save_dir = os.path.join(save_dir, "weights")
        if main_device:
            os.makedirs(ckpt_save_dir, exist_ok=True)  # save checkpoint path

        # to be compatible with old interface
        has_eval_mask = list(isinstance(c, EvalWhileTrain) for c in self.callback)
        if run_eval and not any(has_eval_mask):
            self.callback.append(EvalWhileTrain())
        if not run_eval and any(has_eval_mask):
            ind = has_eval_mask.index(True)
            self.callback.pop(ind)

        # Grad Accumulate
        self.accumulate_cur_step = 0
        self.accumulate_grads = None
        self.accumulate = accumulate
        self.accumulate_grads_fn = self._get_accumulate_grads_fn()

        # Set Checkpoint Manager
        manager = CheckpointManager(ckpt_save_policy="latest_k")
        manager_ema = CheckpointManager(ckpt_save_policy="latest_k") if self.ema else None

        loader = self.dataloader.create_dict_iterator(output_numpy=False, num_epochs=1)
        s_step_time = time.time()
        s_epoch_time = time.time()
        run_context = RunContext(
            epoch_num=epochs,
            steps_per_epoch=self.steps_per_epoch,
            total_steps=self.dataloader.dataset_size,
            trainer=self,
            test_fn=test_fn,
            enable_modelarts=enable_modelarts,
            ckpt_save_dir=ckpt_save_dir,
            save_dir=save_dir,
            train_url=train_url,
            overflow_still_update=overflow_still_update,
            ms_jit=ms_jit,
            rank_size=rank_size,
        )
        self._on_train_begin(run_context)
        for i, data in enumerate(loader):
            cur_epoch = (i // self.steps_per_epoch) + 1
            cur_step = (i % self.steps_per_epoch) + 1
            run_context.cur_epoch_index = cur_epoch
            run_context.cur_step_index = cur_step

            if cur_step == 1:
                self._on_train_epoch_begin(run_context)
            self.global_step += 1
            if self.global_step < warmup_step:
                if warmup_momentum and isinstance(self.optimizer, (nn.SGD, nn.Momentum)):
                    dtype = self.optimizer.momentum.dtype
                    self.optimizer.momentum = Tensor(warmup_momentum[i], dtype)

            imgs, labels = data["image"], data["labels"]
            self._on_train_step_begin(run_context)
            run_context.loss, run_context.lr = self.train_step(imgs, labels, cur_step=cur_step,cur_epoch=cur_epoch)
            self._on_train_step_end(run_context)

            # train log
            if cur_step % self.log_interval == 0:
                logger.info(
                    f"Epoch {cur_epoch}/{epochs}, Step {cur_step}/{self.steps_per_epoch}, "
                    f"step time: {(time.time() - s_step_time) * 1000 / self.log_interval:.2f} ms"
                )
                s_step_time = time.time()

            # save checkpoint per epoch on main device
            if self.main_device and (i + 1) % self.steps_per_epoch == 0:
                # Save Checkpoint
                ms.save_checkpoint(
                    self.optimizer, os.path.join(ckpt_save_dir, f"optim_{self.model_name}.ckpt"), async_save=True
                )
                save_path = os.path.join(ckpt_save_dir, f"{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                manager.save_ckpoint(self.network, num_ckpt=keep_checkpoint_max, save_path=save_path)
                if self.ema:
                    save_path_ema = os.path.join(
                        ckpt_save_dir, f"EMA_{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt"
                    )
                    manager_ema.save_ckpoint(self.ema.ema, num_ckpt=keep_checkpoint_max, save_path=save_path_ema)
                logger.info(f"Saving model to {save_path}")

                if enable_modelarts:
                    sync_data(save_path, train_url + "/weights/" + save_path.split("/")[-1])
                    if self.ema:
                        sync_data(save_path_ema, train_url + "/weights/" + save_path_ema.split("/")[-1])

                logger.info(f"Epoch {cur_epoch}/{epochs}, epoch time: {(time.time() - s_epoch_time) / 60:.2f} min.")
                s_step_time = time.time()
                s_epoch_time = time.time()
            if self.profiler and self.profiler_step_num == cur_step:
                break
            if cur_step == self.steps_per_epoch:
                self._on_train_epoch_end(run_context)

        self._on_train_end(run_context)
        logger.info("End Train.")

    def train_with_datasink(
        self,
        epochs: int,
        main_device: bool,
        warmup_epoch: int = 0,
        warmup_momentum: Union[list, None] = None,
        keep_checkpoint_max: int = 10,
        log_interval: int = 1,
        loss_item_name: list = [],
        save_dir: str = "",
        enable_modelarts: bool = False,
        train_url: str = "",
        run_eval: bool = False,
        test_fn: types.FunctionType = None,
        overflow_still_update: bool = False,
        ms_jit: bool = True,
        rank_size: int = 8,
        profiler_step_num: int = 1
    ):
        # Modify dataset columns name for data sink mode, because dataloader could not send string data to device.
        loader = self.dataloader.project(["image", "labels"])

        # to be compatible with old interface
        has_eval_mask = list(isinstance(c, EvalWhileTrain) for c in self.callback)
        if run_eval and not any(has_eval_mask):
            self.callback.append(EvalWhileTrain())
        if not run_eval and any(has_eval_mask):
            ind = has_eval_mask.index(True)
            self.callback.pop(ind)

        # Change warmup_momentum, list of step -> list of epoch
        warmup_momentum = (
            [warmup_momentum[_i * self.steps_per_epoch] for _i in range(warmup_epoch)]
            + [warmup_momentum[-1], ] * (epochs - warmup_epoch) if warmup_momentum else None
        )

        # Build train epoch func with sink process
        train_epoch_fn = ms.train.data_sink(
            fn=self.train_step_fn,
            dataset=loader,
            sink_size=self.steps_per_epoch,
            jit_config=ms.JitConfig()
        )

        # Attr
        self.epochs = epochs
        self.main_device = main_device
        self.log_interval = log_interval
        self.loss_item_name = loss_item_name
        self.profiler_step_num = profiler_step_num

        # Directories
        ckpt_save_dir = os.path.join(save_dir, "weights")

        if main_device:
            os.makedirs(ckpt_save_dir, exist_ok=True)  # save checkpoint path

        # Set Checkpoint Manager
        manager = CheckpointManager(ckpt_save_policy="latest_k")
        manager_ema = CheckpointManager(ckpt_save_policy="latest_k") if self.ema else None

        run_context = RunContext(
            epoch_num=epochs,
            steps_per_epoch=self.steps_per_epoch,
            total_steps=self.dataloader.dataset_size,
            trainer=self,
            test_fn=test_fn,
            enable_modelarts=enable_modelarts,
            ckpt_save_dir=ckpt_save_dir,
            save_dir=save_dir,
            train_url=train_url,
            overflow_still_update=overflow_still_update,
            ms_jit=ms_jit,
            rank_size=rank_size,
        )

        s_epoch_time = time.time()
        self._on_train_begin(run_context)
        for epoch in range(epochs):
            cur_epoch = epoch + 1
            self.global_step += self.steps_per_epoch
            run_context.cur_epoch_index = cur_epoch
            if epoch == 0:
                logger.warning("In the data sink mode, log output will only occur once each epoch is completed.")
                logger.warning(
                    "The first epoch will be compiled for the graph, which may take a long time; "
                    "You can come back later :)."
                )

            if warmup_momentum and isinstance(self.optimizer, (nn.SGD, nn.Momentum)):
                dtype = self.optimizer.momentum.dtype
                self.optimizer.momentum = Tensor(warmup_momentum[epoch], dtype)

            # train one epoch with datasink
            self._on_train_epoch_begin(run_context)
            _, loss_item, _, _ = train_epoch_fn()

            # print loss and lr
            log_string = f"Epoch {cur_epoch}/{epochs}, Step {self.steps_per_epoch}/{self.steps_per_epoch}"
            if len(self.loss_item_name) < len(loss_item):
                self.loss_item_name += [f"loss_item{i}" for i in range(len(loss_item) - len(self.loss_item_name))]
            for i in range(len(loss_item)):
                log_string += f", {self.loss_item_name[i]}: {loss_item[i].asnumpy():.4f}"
            if self.optimizer.dynamic_lr:
                if self.optimizer.is_group_lr:
                    lr_cell = self.optimizer.learning_rate[0]
                    cur_lr = lr_cell(Tensor(self.global_step, ms.int32)).asnumpy().item()
                else:
                    cur_lr = self.optimizer.learning_rate(Tensor(self.global_step, ms.int32)).asnumpy().item()
            else:
                cur_lr = self.optimizer.learning_rate.asnumpy().item()
            log_string += f", cur_lr: {cur_lr}"
            logger.info(log_string)
            run_context.loss, run_context.lr = loss_item, cur_lr
            self._on_train_epoch_end(run_context)

            # save checkpoint per epoch on main device
            if self.main_device:
                # Save Checkpoint
                ms.save_checkpoint(
                    self.optimizer, os.path.join(ckpt_save_dir, f"optim_{self.model_name}.ckpt"), async_save=True
                )
                save_path = os.path.join(ckpt_save_dir, f"{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                manager.save_ckpoint(self.network, num_ckpt=keep_checkpoint_max, save_path=save_path)
                if self.ema:
                    save_path_ema = os.path.join(
                        ckpt_save_dir, f"EMA_{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt"
                    )
                    manager_ema.save_ckpoint(self.ema.ema, num_ckpt=keep_checkpoint_max, save_path=save_path_ema)
                logger.info(f"Saving model to {save_path}")

                if enable_modelarts:
                    sync_data(save_path, train_url + "/weights/" + save_path.split("/")[-1])
                    if self.ema:
                        sync_data(save_path_ema, train_url + "/weights/" + save_path_ema.split("/")[-1])

                logger.info(f"Epoch {cur_epoch}/{epochs}, epoch time: {(time.time() - s_epoch_time) / 60:.2f} min.")
                s_epoch_time = time.time()

            if self.profiler and math.ceil(self.profiler_step_num/self.steps_per_epoch) == cur_epoch:
                break
        self._on_train_end(run_context)
        logger.info("End Train.")

    def train_step(self, imgs, labels, cur_step=0, cur_epoch=0):
        if self.accumulate == 1:
            loss, loss_item, _, grads_finite = self.train_step_fn(imgs, labels, True)
            self.scaler.adjust(grads_finite)
            if not grads_finite and (cur_step % self.log_interval == 0):
                if self.overflow_still_update:
                    logger.warning(f"overflow, still update, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
                else:
                    logger.warning(f"overflow, drop step, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
        else:
            loss, loss_item, grads, grads_finite = self.train_step_fn(imgs, labels, False)
            self.scaler.adjust(grads_finite)
            if grads_finite or self.overflow_still_update:
                self.accumulate_cur_step += 1
                if self.accumulate_grads:
                    self.accumulate_grads = self.accumulate_grads_fn(
                        self.accumulate_grads, grads
                    )  # update self.accumulate_grads
                else:
                    self.accumulate_grads = grads

                if self.accumulate_cur_step % self.accumulate == 0:
                    self.optimizer(self.accumulate_grads)
                    if self.ema:
                        self.ema.update()
                    logger.info(
                        f"Epoch {cur_epoch}/{self.epochs}, Step {cur_step}/{self.steps_per_epoch}, "
                        f"accumulate: {self.accumulate}, optimizer an accumulate step success."
                    )
                    from mindspore.amp import all_finite

                    if not all_finite(self.accumulate_grads):
                        logger.warning(f"overflow, still update.")
                    # reset accumulate
                    self.accumulate_grads, self.accumulate_cur_step = None, 0
            else:
                logger.warning(
                    f"Epoch {cur_epoch}/{self.epochs}, Step {cur_step}/{self.steps_per_epoch}, "
                    f"accumulate: {self.accumulate}, this step grad overflow, drop. "
                    f"Loss scale adjust to {self.scaler.scale_value.asnumpy()}"
                )

        # train log
        cur_lr = 0
        if cur_step % self.log_interval == 0:
            log_string = (
                f"Epoch {cur_epoch}/{self.epochs}, Step {cur_step}/{self.steps_per_epoch}, imgsize {imgs.shape[2:]}"
            )
            # print loss
            if len(self.loss_item_name) < len(loss_item):
                self.loss_item_name += [f"loss_item{i}" for i in range(len(loss_item) - len(self.loss_item_name))]
            for i in range(len(loss_item)):
                log_string += f", {self.loss_item_name[i]}: {loss_item[i].asnumpy():.4f}"

            # print lr
            if self.optimizer.dynamic_lr:
                if self.optimizer.is_group_lr:
                    lr_cell = self.optimizer.learning_rate[0]
                    cur_lr = lr_cell(Tensor(self.global_step, ms.int32)).asnumpy().item()
                else:
                    cur_lr = self.optimizer.learning_rate(Tensor(self.global_step, ms.int32)).asnumpy().item()
            else:
                cur_lr = self.optimizer.learning_rate.asnumpy().item()
            log_string += f", cur_lr: {cur_lr}"
            logger.info(log_string)
        return loss_item, cur_lr

    def _get_accumulate_grads_fn(self):
        hyper_map = ops.HyperMap()

        def accu_fn(g1, g2):
            g1 = g1 + g2
            return g1

        def accumulate_grads_fn(accumulate_grads, grads):
            success = hyper_map(accu_fn, accumulate_grads, grads)
            return success

        return accumulate_grads_fn

    def _get_transform_stage(self, cur_epoch, stage_epochs=[]):
        _cur_stage = 0
        for _i in range(len(stage_epochs)):
            if cur_epoch <= stage_epochs[_i]:
                _cur_stage = _i
            else:
                break
        return _cur_stage

    def _on_train_begin(self, run_context: RunContext):
        """hooks to run on the beginning of training process"""

        # check callback type validation
        callback = self.callback
        if callback is None:
            callback = []
        assert isinstance(callback, (tuple, list)), (
            f"expect callback to be list of tuple, " f"but got {type(callback)} instead"
        )
        for cb in callback:
            assert isinstance(cb, BaseCallback), (
                f"expect callback element to be subclass of BaseCallback, " f"but got {type(cb)} instead"
            )
        # log callback base info
        logger.info(f"got {len(callback)} active callback as follows:")
        for cb in self.callback:
            logger.info(cb)

        # check range of log interval
        if self.log_interval > self.steps_per_epoch:
            logger.warning(
                f"log interval should be less than total steps of one epoch, "
                f"but got {self.log_interval} > {self.steps_per_epoch}, set log_interval as steps_per_epoch "
                f"{self.steps_per_epoch}"
            )
            self.log_interval = self.steps_per_epoch

        # throw warning of long time cost
        logger.warning(
            "The first epoch will be compiled for the graph, which may take a long time; " "You can come back later :)."
        )

        # execute customized callback
        for cb in self.callback:
            cb.on_train_begin(run_context)

    def _on_train_end(self, run_context: RunContext):
        """hooks to run on the end of training process"""
        for cb in self.callback:
            cb.on_train_end(run_context)

    def _on_train_epoch_begin(self, run_context: RunContext):
        """hooks to run on the beginning of a training epoch"""
        for cb in self.callback:
            cb.on_train_epoch_begin(run_context)

    def _on_train_epoch_end(self, run_context: RunContext):
        """hooks to run on the end of a training epoch"""
        for cb in self.callback:
            cb.on_train_epoch_end(run_context)

    def _on_train_step_begin(self, run_context: RunContext):
        """hooks to run on the beginning of a training step"""
        for cb in self.callback:
            cb.on_train_step_begin(run_context)

    def _on_train_step_end(self, run_context: RunContext):
        """hooks to run on the end of a training step"""
        for cb in self.callback:
            cb.on_train_step_end(run_context)
