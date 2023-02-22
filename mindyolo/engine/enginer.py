import math
import random
import time
import os
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.amp import auto_mixed_precision

from mindyolo.models import create_loss, create_model
from mindyolo.optim import create_group_param, create_lr_scheduler, create_warmup_momentum_scheduler, \
    create_optimizer, EMA
from mindyolo.data import create_dataloader
from mindyolo.data.general import coco80_to_coco91_class
from mindyolo.utils import logger
from mindyolo.utils.all_finite import all_finite
from mindyolo.utils.checkpoint_manager import CheckpointManager
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.modelarts import sync_data

from .env import init_env, set_seed

__all__ = ['Enginer']


class Enginer:
    def __init__(self, cfg, task='train'):

        # Check task
        assert task.lower() in ('train', 'eval', 'test', 'predict'), \
            "Trainer task should be 'train', 'eval', 'test' or 'predict'"

        # Init
        set_seed(cfg.get('seed', 2))
        init_env(cfg)
        self.cfg = cfg
        self.task = task.lower()
        self.img_size = cfg.img_size
        self.rank_size = cfg.rank_size
        self.main_device = cfg.main_device
        self.amp_level = cfg.get('ms_amp_level', 'O0')
        self.ms_jit = cfg.get('ms_jit', False)
        self.is_parallel = cfg.is_parallel = cfg.get('is_parallel', False)
        self.run_eval = cfg.get('run_eval', False)
        self.auto_accumulate = cfg.auto_accumulate
        self.accumulate = cfg.accumulate
        self.multi_scale = cfg.get('multi_scale', False)
        self.overflow_still_update = cfg.get('overflow_still_update', True)
        self.epochs = cfg.epochs
        self.warmup_epochs = cfg.optimizer.warmup_epochs
        self.min_warmup_step = cfg.optimizer.min_warmup_step
        self.stride = cfg.network.stride
        self.log_interval = cfg.get('log_interval', 1)
        self.loss_item_name = cfg.loss.loss_item_name

        # Create Network
        self.network = create_model(model_name=cfg.network.model_name,
                                    model_cfg=cfg.network,
                                    num_classes=cfg.data.nc,
                                    sync_bn=cfg.sync_bn if hasattr(cfg, 'sync_bn') else False)
        if cfg.ema and self.main_device:
            ema_network = create_model(model_name=cfg.network.model_name,
                                       model_cfg=cfg.network,
                                       num_classes=cfg.data.nc)
            self.ema = EMA(self.network, ema_network)
        else:
            self.ema = None
        self.load_pretrain() # load pretrain
        self.freeze_layers() # freeze Layers
        auto_mixed_precision(self.network, amp_level=self.amp_level)
        if self.ema:
            auto_mixed_precision(self.ema.ema, amp_level=self.amp_level)

        if task == 'train':
            # Create Dataset
            self.dataset = create_dataloader(data_config=cfg.data,
                                             task=task,
                                             per_batch_size=cfg.per_batch_size)
            self.steps_per_epoch = self.dataset.get_dataset_size()
            if self.run_eval:
                self.eval_dataset = create_dataloader(data_config=cfg.data,
                                                      task='eval',
                                                      per_batch_size=cfg.per_batch_size * 2)

            # Create Loss
            self.loss = create_loss(
                **cfg.loss,
                anchors=cfg.network.get('anchors', None),
                stride=cfg.network.get('stride', None),
                nc=cfg.data.get('nc', None)
            )
            auto_mixed_precision(self.loss, amp_level=self.amp_level)

            # Create Optimizer
            cfg.optimizer.steps_per_epoch = self.steps_per_epoch
            lr = create_lr_scheduler(**cfg.optimizer)
            params = create_group_param(params=self.network.trainable_params(), **cfg.optimizer)
            self.optimizer = create_optimizer(params=params, lr=lr, **cfg.optimizer)
            self.warmup_momentum = create_warmup_momentum_scheduler(**cfg.optimizer)

            # Create train_step_fn
            self.reducer = self._get_gradreducer()
            self.scaler = self._get_loss_scaler()
            self.train_step_fn = self._get_train_step_fn(network=self.network, loss_fn=self.loss, optimizer=self.optimizer,
                                                         rank_size=self.rank_size, scaler=self.scaler, reducer=self.reducer,
                                                         overflow_still_update=self.overflow_still_update,
                                                         ms_jit=self.ms_jit)
            self.network.set_train(True)
            self.optimizer.set_train(True)

        elif task in ('eval', 'test'):
            self.dataset = create_dataloader(data_config=cfg.data,
                                             task=task,
                                             per_batch_size=cfg.per_batch_size)
            self.network.set_train(False)

        elif task in ('predict', 'detect'):
            raise NotImplementedError

        else:
            raise NotImplementedError

    def train(self):
        self.global_step = 0
        self.accumulate_cur_step = 0
        self.accumulate_grads = None
        self.warmup_steps = max(self.warmup_epochs * self.steps_per_epoch, self.min_warmup_step)
        self.input_dtype = ms.float32 if self.amp_level == "O0" else ms.float16
        ckpt_save_dir = self.cfg.ckpt_save_dir
        keep_checkpoint_max = self.cfg.keep_checkpoint_max
        enable_modelarts = self.cfg.enable_modelarts
        sync_lock_dir = self.cfg.sync_lock_dir
        model_name = os.path.basename(self.cfg.config)[:-5]  # delete ".yaml"
        manager = CheckpointManager(ckpt_save_policy='latest_k')
        manager_ema = CheckpointManager(ckpt_save_policy='latest_k') if self.ema else None
        manager_best = CheckpointManager(ckpt_save_policy='top_k') if self.run_eval else None
        ckpt_filelist_best = []

        s_time = time.time()
        for i in range(self.epochs):
            cur_epoch = i + 1

            self.train_epoch(cur_epoch)

            if self.run_eval:
                s_eval_time = time.time()
                sync_lock = os.path.join(sync_lock_dir, "/run_eval_sync.lock" + str(cur_epoch))
                # single device run eval only
                if self.main_device and not os.path.exists(sync_lock):
                    eval_network = self.ema.ema if self.ema else self.network
                    _train_status = eval_network.training
                    eval_network.set_train(False)
                    accuracy = self.eval(eval_network, self.eval_dataset)
                    accuracy = accuracy[0] if isinstance(accuracy, (list, tuple)) else accuracy
                    eval_network.set_train(_train_status)

                    save_path_best = os.path.join(ckpt_save_dir, f"best/{model_name}-{cur_epoch}_{self.steps_per_epoch}"
                                                                 f"_acc{accuracy:.2f}.ckpt")
                    ckpt_filelist_best = manager_best.save_ckpoint(eval_network, num_ckpt=keep_checkpoint_max,
                                                                   metric=accuracy, save_path=save_path_best)
                    logger.info(f"Epoch {self.epochs}/{cur_epoch}, eval accuracy: {accuracy:.2f}, "
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

            # Each server contains 8 devices as most.
            if self.main_device:
                # Save Checkpoint
                ms.save_checkpoint(self.optimizer, os.path.join(ckpt_save_dir, f'optim_{model_name}.ckpt'),
                                   async_save=True)
                save_path = os.path.join(ckpt_save_dir, f"{model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                manager.save_ckpoint(self.network, num_ckpt=keep_checkpoint_max, save_path=save_path)
                if self.ema:
                    save_path_ema = os.path.join(ckpt_save_dir, f"EMA_{model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                    manager_ema.save_ckpoint(self.ema, num_ckpt=keep_checkpoint_max, save_path=save_path_ema)
                logger.info(f"Saving model to {save_path}")

            if enable_modelarts:
                sync_data(save_path, self.cfg.train_url + "/weights/" + save_path.split("/")[-1])
                if self.ema:
                    sync_data(save_path_ema, self.cfg.train_url + "/weights/" + save_path_ema.split("/")[-1])

            logger.info(f"Epoch {self.epochs}/{cur_epoch}, epoch time: {(time.time() - s_time) / 60:.2f} min.")
            s_time = time.time()

        if enable_modelarts and ckpt_filelist_best:
            for p in ckpt_filelist_best:
                sync_data(p, self.cfg.train_url + '/weights/best/' + p.split("/")[-1])

        logger.info("End Train.")

    def train_epoch(self, cur_epoch):
        loader = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
        s_time = time.time()
        for i, data in enumerate(loader):
            cur_step = i + 1

            self.global_step += 1
            if self.global_step < self.warmup_steps:
                xp, fp = [0, self.warmup_steps], [1, self.cfg.nbs / self.cfg.total_batch_size]
                self.accumulate = max(1, np.interp(self.global_step, xp, fp).round()) if self.auto_accumulate else self.accumulate
                if self.warmup_momentum and isinstance(self.optimizer, (nn.SGD, nn.Momentum)):
                    dtype = self.optimizer.momentum.dtype
                    self.optimizer.momentum = Tensor(self.warmup_momentum[i], dtype)

            imgs, batch_idx, gt_class, gt_bbox = data["image"], data['batch_idx'], data["gt_class"], data["gt_bbox"]
            labels = np.concatenate((batch_idx, gt_class, gt_bbox), -1) # (bs, N, 6)
            imgs, labels = Tensor(imgs, self.input_dtype), Tensor(labels, self.input_dtype)
            size = None
            if self.multi_scale:
                gs = max(int(np.array(self.stride).max()), 32)
                sz = random.randrange(self.img_size * 0.5, self.img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    size = tuple([math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]])  # new shape (stretched to gs-multiple)

            self.train_step(imgs, labels, size, cur_step=cur_step, cur_epoch=cur_epoch)

            # train log
            if cur_step % self.log_interval == 0:
                logger.info(f"Epoch {self.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, "
                            f"step time: {(time.time() - s_time) * 1000 / self.log_interval:.2f} ms")
                s_time = time.time()

    def train_step(self, imgs, labels, size=None, cur_step=0, cur_epoch=0):
        if self.accumulate == 1:
            loss, loss_item, _, grads_finite = self.train_step_fn(imgs, labels, size, True)
            if self.ema:
                self.ema.update()
            self.scaler.adjust(grads_finite)
            if not grads_finite:
                if self.overflow_still_update:
                    logger.warning(f"overflow, still update, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
                else:
                    logger.warning(f"overflow, drop step, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
        else:
            loss, loss_item, grads, grads_finite = self.train_step_fn(imgs, labels, size, False)
            self.scaler.adjust(grads_finite)
            if grads_finite or self.overflow_still_update:
                self.accumulate_cur_step += 1
                if self.accumulate_grads:
                    self.accumulate_grads = self.accumulate_grads_fn(self.accumulate_grads, grads)
                else:
                    self.accumulate_grads = grads

                if self.accumulate_cur_step % self.accumulate == 0:
                    self.optimizer(self.accumulate_grads)
                    if self.ema:
                        self.ema.update()
                        logger.info(f"-Epoch: {cur_epoch}, Step: {cur_step}, accumulate: {self.accumulate}, "
                                    f"optimizer an accumulate step success.")
                    if not all_finite(self.accumulate_grads):
                        logger.warning(f"overflow, still update.")
                    # reset accumulate
                    self.accumulate_grads, self.accumulate_cur_step = None, 0
            else:
                logger.warning(f"Epoch: {cur_epoch}, Step: {cur_step}, accumulate: {self.accumulate}, "
                               f"this step grad overflow, drop. Loss scale adjust to {self.scaler.scale_value.asnumpy()}")

        # train log
        if cur_step % self.log_interval == 0:
            size = size if size else imgs.shape[2:]
            log_string = f"Epoch {self.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step + 1}, imgsize {size}"
            # print loss
            if len(self.loss_item_name) < len(loss_item):
                self.loss_item_name += [f'loss_item{i}' for i in range(len(loss_item) - len(self.loss_item_name))]
            for i in range(len(loss_item)):
                log_string += f", {self.loss_item_name[i]}: {loss_item[i].asnumpy():.4f}"
            # print lr
            if self.optimizer.dynamic_lr:
                if self.optimizer.is_group_lr:
                    cur_lr = [lr_cell(Tensor(self.global_step, ms.int32)).asnumpy().item() \
                              for lr_cell in self.optimizer.learning_rate]
                else:
                    cur_lr = self.optimizer.learning_rate(Tensor(self.global_step, ms.int32)).asnumpy().item()
            else:
                cur_lr = self.optimizer.learning_rate.asnumpy().item()
            log_string += f", cur_lr: {cur_lr}"
            logger.info(log_string)

    def eval(self, model=None, dataset=None):

        model = model if model else self.network
        dataset = dataset if dataset else self.dataset
        loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
        anno_json_path = os.path.join(self.cfg.data.dataset_dir, self.cfg.data.val_anno_path)
        coco91class = coco80_to_coco91_class()
        is_coco_dataset = ('coco' in self.cfg.data.dataset_name)

        step_num = dataset.get_dataset_size()
        infer_times = 0.
        nms_times = 0.
        result_dicts = []

        for i, data in enumerate(loader):
            imgs, batch_idx, gt_class, gt_bbox, paths, ori_shape = \
                data["image"], data['batch_idx'], data["gt_class"], data["gt_bbox"],\
                data["im_file"], data["ori_shape"]
            imgs_tensor = Tensor(imgs, self.input_dtype)
            nb, _, height, width = imgs.shape  # batch size, channels, height, width
            targets = np.concatenate((batch_idx, gt_class, gt_bbox), -1)  # (bs, N, 6)
            targets = targets.reshape((-1, 6))
            targets = targets[targets[:, 1] >= 0]

            # Run infer
            _t = time.time()
            out = model(imgs_tensor)  # inference and training outputs
            out = out[0] if isinstance(out, (tuple, list)) else out
            infer_times += time.time() - _t

            # Run NMS
            targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
            t = time.time()
            out = out.asnumpy()
            out = non_max_suppression(out, conf_thres=self.cfg.conf_thres, iou_thres=self.cfg.iou_thres,
                                       multi_label=True, time_limit=self.cfg.nms_time_limit)
            nms_times += time.time() - t

            # Statistics pred
            for si, pred in enumerate(out):
                path = Path(str(paths[si]))
                if len(pred) == 0:
                    continue

                # Predictions
                predn = np.copy(pred)
                scale_coords(imgs[si].shape[1:], predn[:, :4], ori_shape[si])  # native-space pred

                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    result_dicts.append({'image_id': image_id,
                                         'category_id': coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                                         'bbox': [round(x, 3) for x in b],
                                         'score': round(p[4], 5)})

        # Compute mAP
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            anno = COCO(anno_json_path)  # init annotations api
            pred = anno.loadRes(result_dicts)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco_dataset:
                eval.params.imgIds = [int(Path(img_rec['im_file']).stem) for img_rec in loader.source.imgs_records] # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
            raise e

        t = tuple(x / step_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
            (height, width, self.cfg.per_batch_size)  # tuple
        print(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

        return map, map50

    def export(self):
        # TODO: Supplement export function
        pass

    def predict(self):
        # TODO: Supplement predict function
        pass

    def load_pretrain(self):
        weight, ema_weight = self.cfg.weight, self.cfg.ema_weight

        if weight.endswith('.ckpt'):
            param_dict = ms.load_checkpoint(weight)
            ms.load_param_into_net(self.network, param_dict)
            logger.info(f"Pretrain model load from \"{weight}\" success.")
        if self.ema:
            if ema_weight.endswith('.ckpt'):
                param_dict_ema = ms.load_checkpoint(ema_weight)
                ms.load_param_into_net(self.ema.ema, param_dict_ema)
                logger.info(f"Ema pretrain model load from \"{ema_weight}\" success.")
            else:
                self.ema.clone_from_model()
                logger.info("ema_weight not exist, default pretrain weight is currently used.")

    def freeze_layers(self):
        freeze = self.cfg.freeze
        if len(freeze) > 0:
            freeze = [f'model.{x}.' for x in freeze]  # parameter names to freeze (full or partial)
            for n, p in self.network.parameters_and_names():
                if any(x in n for x in freeze):
                    print('freezing %s' % n)
                    p.requires_grad = False

    @staticmethod
    @ms.ms_function
    def accumulate_grads_fn(accumulate_grads, grads):

        def accu_fn(g1, g2):
            return g1 + g2

        return ops.HyperMap()(accu_fn, accumulate_grads, grads)

    @staticmethod
    def _get_train_step_fn(network, loss_fn, optimizer, rank_size, scaler, reducer, overflow_still_update=False, ms_jit=False):

        def forward_func(x, label, sizes=None):
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = network(x)
            loss, loss_items = loss_fn(pred, label, x)
            loss *= rank_size
            return scaler.scale(loss), loss_items

        grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

        def train_step_func(x, label, sizes=None, optimizer_update=True):
            (loss, loss_items), grads = grad_fn(x, label, sizes)
            loss = scaler.unscale(loss)
            grads = reducer(grads)
            unscaled_grads = scaler.unscale(grads)
            grads_finite = all_finite(unscaled_grads)

            if optimizer_update:
                if grads_finite:
                    loss = ops.depend(loss, optimizer(unscaled_grads))
                else:
                    if overflow_still_update:
                        loss = ops.depend(loss, optimizer(unscaled_grads))

            return loss, loss_items, unscaled_grads, grads_finite

        @ms.ms_function
        def jit_warpper(*args):
            return train_step_func(*args)

        return train_step_func if not ms_jit else jit_warpper

    def _get_gradreducer(self):
        if self.is_parallel:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters, mean, degree)
        else:
            grad_reducer = ops.functional.identity

        return grad_reducer

    def _get_loss_scaler(self):
        ms_loss_scaler = self.cfg.ms_loss_scaler
        if ms_loss_scaler == 'dynamic':
            from mindspore.amp import DynamicLossScaler
            loss_scaler = DynamicLossScaler(scale_value=self.cfg.get('ms_loss_scaler_value', 2 ** 16),
                                            scale_factor=self.cfg.get('scale_factor', 2),
                                            scale_window=self.cfg.get('scale_window', 2000))
        elif ms_loss_scaler == 'static':
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(self.cfg.get('ms_loss_scaler_value', 2 ** 10))
        elif ms_loss_scaler in ('none', 'None'):
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(1.0)
        else:
            raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

        return loss_scaler
