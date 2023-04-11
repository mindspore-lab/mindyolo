import os
import types
import time
from typing import Union

import mindspore as ms
from mindspore import nn, ops, Tensor

from mindyolo.utils import logger
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
    ema: nn.Cell,
    optimizer: nn.Cell,
    dataloader: ms.dataset.Dataset,
):
    return Trainer(
        model_name=model_name, train_step_fn=train_step_fn, scaler=scaler,
        dataloader=dataloader,
        network=network, ema=ema, optimizer=optimizer
    )


class Trainer:
    def __init__(self, model_name, train_step_fn, scaler, network, ema, optimizer, dataloader):
        self.model_name = model_name
        self.train_step_fn = train_step_fn
        self.scaler = scaler
        self.dataloader = dataloader
        self.network = network      # for save checkpoint
        self.ema = ema              # for save checkpoint
        self.optimizer = optimizer  # for save checkpoint
        self.global_step = 0
        self.steps_per_epoch = self.dataloader.get_dataset_size()

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
            save_dir: str = '',
            enable_modelarts: bool = False,
            train_url: str = '',
            run_eval: bool = False,
            test_fn: types.FunctionType = None,
    ):
        # Set Attr
        self.epochs = epochs
        self.main_device = main_device
        self.log_interval = log_interval
        self.overflow_still_update = overflow_still_update
        self.loss_item_name = loss_item_name

        # Directories settings
        ckpt_save_dir = os.path.join(save_dir, 'weights')
        sync_lock_dir = os.path.join(save_dir, 'sync_locks') if not enable_modelarts else '/tmp/sync_locks'
        if main_device:
            os.makedirs(ckpt_save_dir, exist_ok=True)   # save checkpoint path
            os.makedirs(sync_lock_dir, exist_ok=False)  # sync_lock for run_eval

        # Grad Accumulate
        self.accumulate_cur_step = 0
        self.accumulate_grads = None
        self.accumulate = accumulate
        self.accumulate_grads_fn = self._get_accumulate_grads_fn()

        manager = CheckpointManager(ckpt_save_policy='latest_k')
        manager_ema = CheckpointManager(ckpt_save_policy='latest_k') if self.ema else None
        manager_best = CheckpointManager(ckpt_save_policy='top_k') if run_eval else None
        ckpt_filelist_best = []

        self.dataloader = self.dataloader.repeat(epochs)
        loader = self.dataloader.create_dict_iterator(output_numpy=False, num_epochs=1)
        s_step_time = time.time()
        s_epoch_time = time.time()
        for i, data in enumerate(loader):
            cur_epoch = (i // self.steps_per_epoch) + 1
            cur_step = (i % self.steps_per_epoch) + 1

            self.global_step += 1
            if self.global_step < warmup_step:
                if warmup_momentum and isinstance(self.optimizer, (nn.SGD, nn.Momentum)):
                    dtype = self.optimizer.momentum.dtype
                    self.optimizer.momentum = Tensor(warmup_momentum[i], dtype)

            imgs, labels = data['image'], data['labels']
            self.train_step(imgs, labels, cur_step=cur_step, cur_epoch=cur_epoch)

            # train log
            if cur_step % self.log_interval == 0:
                logger.info(f"Epoch {epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, "
                            f"step time: {(time.time() - s_step_time) * 1000 / self.log_interval:.2f} ms")
                s_step_time = time.time()

            # run eval per epoch on main device
            if run_eval and (i + 1) % self.steps_per_epoch == 0:
                s_eval_time = time.time()
                sync_lock = os.path.join(sync_lock_dir, "/run_eval_sync.lock" + str(cur_epoch))
                # single device run eval only
                if self.main_device and not os.path.exists(sync_lock):
                    eval_network = self.ema.ema if self.ema else self.network
                    _train_status = eval_network.training
                    eval_network.set_train(False)
                    accuracy = test_fn(network=eval_network)
                    accuracy = accuracy[0] if isinstance(accuracy, (list, tuple)) else accuracy
                    eval_network.set_train(_train_status)

                    save_path_best = os.path.join(ckpt_save_dir, f"best/{self.model_name}-{cur_epoch}_{self.steps_per_epoch}"
                                                                 f"_acc{accuracy:.2f}.ckpt")
                    ckpt_filelist_best = manager_best.save_ckpoint(eval_network, num_ckpt=keep_checkpoint_max,
                                                                   metric=accuracy, save_path=save_path_best)
                    logger.info(f"Epoch {epochs}/{cur_epoch}, eval accuracy: {accuracy:.2f}, "
                                f"run_eval time: {(time.time() - s_eval_time):.2f} s.")
                    try:
                        os.mknod(sync_lock)
                    except IOError:
                        pass
                # other device wait for lock sign
                while True:
                    if os.path.exists(sync_lock):
                        break
                    time.sleep(1)

            # save checkpoint per epoch on main device
            if self.main_device and (i + 1) % self.steps_per_epoch == 0:
                # Save Checkpoint
                ms.save_checkpoint(self.optimizer, os.path.join(ckpt_save_dir, f'optim_{self.model_name}.ckpt'),
                                   async_save=True)
                save_path = os.path.join(ckpt_save_dir, f"{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                manager.save_ckpoint(self.network, num_ckpt=keep_checkpoint_max, save_path=save_path)
                if self.ema:
                    save_path_ema = os.path.join(ckpt_save_dir,
                                                 f"EMA_{self.model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                    manager_ema.save_ckpoint(self.ema.ema, num_ckpt=keep_checkpoint_max, save_path=save_path_ema)
                logger.info(f"Saving model to {save_path}")

                if enable_modelarts:
                    sync_data(save_path, train_url + "/weights/" + save_path.split("/")[-1])
                    if self.ema:
                        sync_data(save_path_ema, train_url + "/weights/" + save_path_ema.split("/")[-1])

                logger.info(
                    f"Epoch {epochs}/{cur_epoch}, epoch time: {(time.time() - s_epoch_time) / 60:.2f} min.")
                s_epoch_time = time.time()

        if enable_modelarts and ckpt_filelist_best:
            for p in ckpt_filelist_best:
                sync_data(p, train_url + '/weights/best/' + p.split("/")[-1])

        logger.info("End Train.")

    def train_step(self, imgs, labels, cur_step=0, cur_epoch=0):
        if self.accumulate == 1:
            loss, loss_item, _, grads_finite = self.train_step_fn(imgs, labels, True)
            if self.ema:
                self.ema.update()
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
                    self.accumulate_grads = self.accumulate_grads_fn(self.accumulate_grads, grads) # update self.accumulate_grads
                else:
                    self.accumulate_grads = grads

                if self.accumulate_cur_step % self.accumulate == 0:
                    self.optimizer(self.accumulate_grads)
                    if self.ema:
                        self.ema.update()
                    logger.info(f"Epoch {self.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, "
                                f"accumulate: {self.accumulate}, optimizer an accumulate step success.")
                    from mindyolo.utils.all_finite import all_finite
                    if not all_finite(self.accumulate_grads):
                        logger.warning(f"overflow, still update.")
                    # reset accumulate
                    self.accumulate_grads, self.accumulate_cur_step = None, 0
            else:
                logger.warning(f"Epoch {self.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, "
                               f"accumulate: {self.accumulate}, this step grad overflow, drop. "
                               f"Loss scale adjust to {self.scaler.scale_value.asnumpy()}")

        # train log
        if cur_step % self.log_interval == 0:
            log_string = f"Epoch {self.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, imgsize {imgs.shape[2:]}"
            # print loss
            if len(self.loss_item_name) < len(loss_item):
                self.loss_item_name += [f'loss_item{i}' for i in range(len(loss_item) - len(self.loss_item_name))]
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

    def _get_accumulate_grads_fn(self):
        hyper_map = ops.HyperMap()

        def accu_fn(g1, g2):
            g1 = g1 + g2
            return g1

        def accumulate_grads_fn(accumulate_grads, grads):
            success = hyper_map(accu_fn, accumulate_grads, grads)
            return success

        return accumulate_grads_fn
