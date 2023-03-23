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

from mindyolo.models import create_loss, create_model
from mindyolo.optim import create_group_param, create_lr_scheduler, create_warmup_momentum_scheduler, \
    create_optimizer, EMA
from mindyolo.data import COCODataset, create_loader
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils import logger
from mindyolo.utils.checkpoint_manager import CheckpointManager
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.modelarts import sync_data

from .utils import set_default, set_seed

__all__ = ['Enginer']


class Enginer:
    def __init__(self, cfg, task='train'):

        # Init
        set_seed(cfg.seed)
        set_default(cfg)
        self.cfg = cfg
        self.main_device = (cfg.rank % cfg.rank_size == 0)
        self.input_dtype = ms.float32 if self.cfg.ms_amp_level == "O0" else ms.float16

        # Create Network
        cfg.network.recompute = cfg.recompute
        cfg.network.recompute_layers = cfg.recompute_layers
        self.network = create_model(model_name=cfg.network.model_name,
                                    model_cfg=cfg.network,
                                    num_classes=cfg.data.nc,
                                    sync_bn=cfg.sync_bn)
        if cfg.ema and self.main_device:
            ema_network = create_model(model_name=cfg.network.model_name,
                                       model_cfg=cfg.network,
                                       num_classes=cfg.data.nc)
            self.ema = EMA(self.network, ema_network)
        else:
            self.ema = None
        self.load_pretrain() # load pretrain
        self.freeze_layers() # freeze Layers
        ms.amp.auto_mixed_precision(self.network, amp_level=self.cfg.ms_amp_level)
        if self.ema:
            ms.amp.auto_mixed_precision(self.ema.ema, amp_level=self.cfg.ms_amp_level)

        if task == 'train':
            # Create Dataset
            self.dataset = COCODataset(
                dataset_dir=cfg.data.dataset_dir,
                image_dir=cfg.data.train_img_dir,
                anno_path=cfg.data.train_anno_path,
                img_size=cfg.img_size,
                transforms_dict=cfg.data.train_transforms,
                is_training=True,
                rect=cfg.rect,
                batch_size=cfg.total_batch_size,
                stride=max(cfg.network.stride),
            )
            self.dataloader = create_loader(
                dataset=self.dataset,
                batch_collate_fn=self.dataset.train_collate_fn,
                dataset_column_names=self.dataset.dataset_column_names,
                batch_size=cfg.per_batch_size,
                epoch_size=1,
                rank=cfg.rank,
                rank_size=cfg.rank_size,
                shuffle=True,
                drop_remainder=True,
                num_parallel_workers=cfg.data.num_parallel_workers,
                python_multiprocessing=True
            )
            self.steps_per_epoch = self.dataloader.get_dataset_size()

            if self.cfg.run_eval:
                self.eval_dataset = COCODataset(
                    dataset_dir=cfg.data.dataset_dir,
                    image_dir=cfg.data.val_img_dir,
                    anno_path=cfg.data.val_anno_path,
                    img_size=cfg.img_size,
                    transforms_dict=cfg.data.test_transforms,
                    is_training=False, rect=False, batch_size=cfg.per_batch_size * 2,
                    stride=max(cfg.network.stride),
                )
                self.eval_dataloader = create_loader(
                    dataset=self.eval_dataset,
                    batch_collate_fn=self.eval_dataset.test_collate_fn,
                    dataset_column_names=self.eval_dataset.dataset_column_names,
                    batch_size=cfg.per_batch_size * 2,
                    epoch_size=1, rank=0, rank_size=1, shuffle=False, drop_remainder=False,
                    num_parallel_workers=cfg.data.num_parallel_workers,
                    python_multiprocessing=True
                )

            # Create Loss
            self.loss = create_loss(
                **cfg.loss,
                anchors=cfg.network.anchors,
                stride=cfg.network.stride,
                nc=cfg.data.nc
            )
            ms.amp.auto_mixed_precision(self.loss, amp_level=self.cfg.ms_amp_level)

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
                                                         rank_size=self.cfg.rank_size, scaler=self.scaler, reducer=self.reducer,
                                                         overflow_still_update=self.cfg.overflow_still_update,
                                                         ms_jit=self.cfg.ms_jit)
            self.accumulate_grads_fn = self.get_accumulate_grads_fn()
            self.network.set_train(True)
            self.optimizer.set_train(True)

        elif task in ('val', 'eval', 'test'):

            if task in ('val', 'eval'):
                image_dir = cfg.data.val_img_dir
                anno_path = cfg.data.val_anno_path
            else:
                image_dir = cfg.data.test_img_dir
                anno_path = cfg.data.test_anno_path

            self.dataset = COCODataset(
                dataset_dir=cfg.data.dataset_dir,
                image_dir=image_dir,
                anno_path=anno_path,
                img_size=cfg.img_size,
                transforms_dict=cfg.data.test_transforms,
                is_training=False, rect=False,
                batch_size=cfg.per_batch_size * 2,
                stride=max(cfg.network.stride),
            )
            self.dataloader = create_loader(
                dataset=self.dataset,
                batch_collate_fn=self.dataset.test_collate_fn,
                dataset_column_names=self.dataset.dataset_column_names,
                batch_size=cfg.per_batch_size * 2,
                epoch_size=1, rank=0, rank_size=1, shuffle=False, drop_remainder=False,
                num_parallel_workers=cfg.data.num_parallel_workers,
                python_multiprocessing=True
            )
            self.network.set_train(False)

        elif task in ('infer', 'detect', 'export', '310'):
            self.network.set_train(False)

        else:
            raise NotImplementedError

    def train(self):
        self.global_step = 0
        self.warmup_steps = max(round(self.cfg.optimizer.warmup_epochs * self.steps_per_epoch),
                                self.cfg.optimizer.min_warmup_step)
        ckpt_save_dir = self.cfg.ckpt_save_dir
        keep_checkpoint_max = self.cfg.keep_checkpoint_max
        enable_modelarts = self.cfg.enable_modelarts
        sync_lock_dir = self.cfg.sync_lock_dir

        # grad accumulate
        self.accumulate_cur_step = 0
        self.accumulate_grads = None
        self.auto_accumulate = self.cfg.auto_accumulate
        self.accumulate = self.cfg.accumulate

        model_name = os.path.basename(self.cfg.config)[:-5]  # delete ".yaml"
        manager = CheckpointManager(ckpt_save_policy='latest_k')
        manager_ema = CheckpointManager(ckpt_save_policy='latest_k') if self.ema else None
        manager_best = CheckpointManager(ckpt_save_policy='top_k') if self.cfg.run_eval else None
        ckpt_filelist_best = []

        self.dataloader = self.dataloader.repeat(self.cfg.epochs)
        loader = self.dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        s_step_time = time.time()
        s_epoch_time = time.time()
        for i, data in enumerate(loader):
            cur_epoch = (i // self.steps_per_epoch) + 1
            cur_step = (i % self.steps_per_epoch) + 1

            self.global_step += 1
            if self.global_step < self.warmup_steps:
                xp, fp = [0, self.warmup_steps], [1, self.cfg.nbs / self.cfg.total_batch_size]
                self.accumulate = max(1, np.interp(self.global_step, xp,
                                                   fp).round()) if self.cfg.auto_accumulate else self.accumulate
                if self.warmup_momentum and isinstance(self.optimizer, (nn.SGD, nn.Momentum)):
                    dtype = self.optimizer.momentum.dtype
                    self.optimizer.momentum = Tensor(self.warmup_momentum[i], dtype)

            imgs, labels = data['image'], data['labels']
            imgs, labels = Tensor(imgs, self.input_dtype), Tensor(labels, self.input_dtype)
            size = None
            if self.cfg.multi_scale:
                gs = max(int(np.array(self.cfg.network.stride).max()), 32)
                sz = random.randrange(self.cfg.img_size * 0.5, self.cfg.img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    size = tuple(
                        [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]])  # new shape (stretched to gs-multiple)

            self.train_step(imgs, labels, size, cur_step=cur_step, cur_epoch=cur_epoch)

            # train log
            if cur_step % self.cfg.log_interval == 0:
                logger.info(f"Epoch {self.cfg.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, "
                            f"step time: {(time.time() - s_step_time) * 1000 / self.cfg.log_interval:.2f} ms")
                s_step_time = time.time()

            # run eval per epoch on main device
            if self.cfg.run_eval and (i + 1) % self.steps_per_epoch == 0:
                s_eval_time = time.time()
                sync_lock = os.path.join(sync_lock_dir, "/run_eval_sync.lock" + str(cur_epoch))
                # single device run eval only
                if self.main_device and not os.path.exists(sync_lock):
                    eval_network = self.ema.ema if self.ema else self.network
                    _train_status = eval_network.training
                    eval_network.set_train(False)
                    accuracy = self.eval(eval_network, self.eval_dataloader, self.eval_dataset)
                    accuracy = accuracy[0] if isinstance(accuracy, (list, tuple)) else accuracy
                    eval_network.set_train(_train_status)

                    save_path_best = os.path.join(ckpt_save_dir, f"best/{model_name}-{cur_epoch}_{self.steps_per_epoch}"
                                                                 f"_acc{accuracy:.2f}.ckpt")
                    ckpt_filelist_best = manager_best.save_ckpoint(eval_network, num_ckpt=keep_checkpoint_max,
                                                                   metric=accuracy, save_path=save_path_best)
                    logger.info(f"Epoch {self.cfg.epochs}/{cur_epoch}, eval accuracy: {accuracy:.2f}, "
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
                ms.save_checkpoint(self.optimizer, os.path.join(ckpt_save_dir, f'optim_{model_name}.ckpt'),
                                   async_save=True)
                save_path = os.path.join(ckpt_save_dir, f"{model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                manager.save_ckpoint(self.network, num_ckpt=keep_checkpoint_max, save_path=save_path)
                if self.ema:
                    save_path_ema = os.path.join(ckpt_save_dir,
                                                 f"EMA_{model_name}-{cur_epoch}_{self.steps_per_epoch}.ckpt")
                    manager_ema.save_ckpoint(self.ema.ema, num_ckpt=keep_checkpoint_max, save_path=save_path_ema)
                logger.info(f"Saving model to {save_path}")

                if enable_modelarts:
                    sync_data(save_path, self.cfg.train_url + "/weights/" + save_path.split("/")[-1])
                    if self.ema:
                        sync_data(save_path_ema, self.cfg.train_url + "/weights/" + save_path_ema.split("/")[-1])

                logger.info(f"Epoch {self.cfg.epochs}/{cur_epoch}, epoch time: {(time.time() - s_epoch_time) / 60:.2f} min.")
                s_epoch_time = time.time()

        if enable_modelarts and ckpt_filelist_best:
            for p in ckpt_filelist_best:
                sync_data(p, self.cfg.train_url + '/weights/best/' + p.split("/")[-1])

        logger.info("End Train.")

    def train_step(self, imgs, labels, size=None, cur_step=0, cur_epoch=0):
        if self.accumulate == 1:
            loss, loss_item, _, grads_finite = self.train_step_fn(imgs, labels, size, True)
            if self.ema:
                self.ema.update()
            self.scaler.adjust(grads_finite)
            if not grads_finite:
                if self.cfg.overflow_still_update:
                    logger.warning(f"overflow, still update, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
                else:
                    logger.warning(f"overflow, drop step, loss scale adjust to {self.scaler.scale_value.asnumpy()}")
        else:
            loss, loss_item, grads, grads_finite = self.train_step_fn(imgs, labels, size, False)
            self.scaler.adjust(grads_finite)
            if grads_finite or self.cfg.overflow_still_update:
                self.accumulate_cur_step += 1
                if self.accumulate_grads:
                    self.accumulate_grads = self.accumulate_grads_fn(self.accumulate_grads, grads) # update self.accumulate_grads
                else:
                    self.accumulate_grads = grads

                if self.accumulate_cur_step % self.accumulate == 0:
                    self.optimizer(self.accumulate_grads)
                    if self.ema:
                        self.ema.update()
                    logger.info(f"Epoch {self.cfg.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, accumulate: {self.accumulate}, "
                                f"optimizer an accumulate step success.")
                    from mindyolo.utils.all_finite import all_finite
                    if not all_finite(self.accumulate_grads):
                        logger.warning(f"overflow, still update.")
                    # reset accumulate
                    self.accumulate_grads, self.accumulate_cur_step = None, 0
            else:
                logger.warning(f"Epoch {self.cfg.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, accumulate: {self.accumulate}, "
                               f"this step grad overflow, drop. Loss scale adjust to {self.scaler.scale_value.asnumpy()}")

        # train log
        if cur_step % self.cfg.log_interval == 0:
            size = size if size else imgs.shape[2:]
            log_string = f"Epoch {self.cfg.epochs}/{cur_epoch}, Step {self.steps_per_epoch}/{cur_step}, imgsize {size}"
            # print loss
            _loss_item_name = self.cfg.loss.loss_item_name
            if len(_loss_item_name) < len(loss_item):
                _loss_item_name += [f'loss_item{i}' for i in range(len(loss_item) - len(_loss_item_name))]
            for i in range(len(loss_item)):
                log_string += f", {_loss_item_name[i]}: {loss_item[i].asnumpy():.4f}"
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

    def eval(self, model=None, dataloader=None, dataset=None):

        model = model if model else self.network
        dataset = dataset if dataset else self.dataset
        dataloader = dataloader if dataloader else self.dataloader
        loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        anno_json_path = os.path.join(self.cfg.data.dataset_dir, self.cfg.data.val_anno_path)
        coco91class = COCO80_TO_COCO91_CLASS
        is_coco_dataset = ('coco' in self.cfg.data.dataset_name)

        step_num = dataloader.get_dataset_size()
        sample_num = 0
        infer_times = 0.
        nms_times = 0.
        result_dicts = []

        for i, data in enumerate(loader):
            imgs, _, paths, ori_shape, pad, hw_scale = data['image'], data['labels'], data['img_files'], \
                                                       data['hw_ori'], data['pad'], data['hw_scale']
            nb, _, height, width = imgs.shape
            imgs_tensor = Tensor(imgs, self.input_dtype)

            # Run infer
            _t = time.time()
            out, _ = model(imgs_tensor)  # inference and training outputs
            # out = out[0] if isinstance(out, (tuple, list)) else out
            infer_times += time.time() - _t

            # Run NMS
            t = time.time()
            out = out.asnumpy()
            out = non_max_suppression(out, conf_thres=self.cfg.conf_thres, iou_thres=self.cfg.iou_thres,
                                      multi_label=True, time_limit=self.cfg.nms_time_limit)
            nms_times += time.time() - t

            # Statistics pred
            for si, pred in enumerate(out):
                path = Path(str(paths[si]))
                sample_num += 1
                if len(pred) == 0:
                    continue

                # Predictions
                predn = np.copy(pred)
                scale_coords(imgs[si].shape[1:], predn[:, :4], ori_shape[si], ratio=hw_scale[si], pad=pad[si])  # native-space pred

                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    result_dicts.append({'image_id': image_id,
                                         'category_id': coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                                         'bbox': [round(x, 3) for x in b],
                                         'score': round(p[4], 5)})
            logger.info(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

        # Compute mAP
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            anno = COCO(anno_json_path)  # init annotations api
            pred = anno.loadRes(result_dicts)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco_dataset:
                eval.params.imgIds = [int(Path(im_file).stem) for im_file in dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            logger.warning(f'pycocotools unable to run: {e}')
            raise e

        t = tuple(x / sample_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
            (height, width, self.cfg.per_batch_size)  # tuple
        logger.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

        return map, map50

    def detect(self, img):
        if isinstance(img, str) and os.path.isfile(img):
            import cv2
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise ValueError("Detect: input image not available.")
        coco91class = COCO80_TO_COCO91_CLASS
        is_coco_dataset = ('coco' in self.cfg.data.dataset_name)

        # Resize
        h_ori, w_ori = img.shape[:2]  # orig hw
        r = self.cfg.img_size / max(h_ori, w_ori)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        h, w = img.shape[:2]
        if h < self.cfg.img_size or w < self.cfg.img_size:
            _stride = self.cfg.network.stride
            new_h, new_w = math.ceil(h / _stride) * _stride, math.ceil(w / _stride) * _stride
            dh, dw = (new_h - h) / 2, (new_w - w) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        # Transpose and Norm
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.
        # To Tensor
        imgs_tensor = Tensor(img[None], self.input_dtype)

        # Run infer
        _t = time.time()
        out = self.network(imgs_tensor)  # inference and training outputs
        out = out[0] if isinstance(out, (tuple, list)) else out
        infer_times = time.time() - _t

        # Run NMS
        t = time.time()
        out = out.asnumpy()
        out = non_max_suppression(out, conf_thres=self.cfg.conf_thres, iou_thres=self.cfg.iou_thres,
                                  multi_label=True, time_limit=self.cfg.nms_time_limit)
        nms_times = time.time() - t

        total_category_ids, total_bboxes, total_scores = [], [], []
        for si, pred in enumerate(out):
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # native-space pred

            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            category_ids, bboxes, scores = [], [], []
            for p, b in zip(pred.tolist(), box.tolist()):
                category_ids.append(coco91class[int(p[5])] if is_coco_dataset else int(p[5]))
                bboxes.append([round(x, 3) for x in b])
                scores.append(round(p[4], 5))

            total_category_ids.extend(category_ids)
            total_bboxes.extend(bboxes)
            total_scores.extend(scores)

        result_dict = {
            'category_id': total_category_ids,
            'bbox': total_bboxes,
            'score': total_scores
        }

        t = tuple(x * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
            (self.cfg.img_size, self.cfg.img_size, 1)  # tuple
        logger.info(f"Predict result is: {result_dict}")
        logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
        logger.info(f"Detect a image success.")

        return result_dict

    def export(self):
        from mindspore import export
        input_arr = Tensor(np.ones([self.cfg.per_batch_size, 3, self.cfg.img_size, self.cfg.img_size]), ms.float32)
        file_name = os.path.basename(self.cfg.config)[:-5]  # delete ".yaml"
        export(self.network, input_arr, file_name=file_name, file_format=self.cfg.file_format)

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
                    logger.info('freezing %s' % n)
                    p.requires_grad = False

    def get_accumulate_grads_fn(self):
        hyper_map = ops.HyperMap()

        def accu_fn(g1, g2):
            g1 = g1 + g2
            return g1

        def accumulate_grads_fn(accumulate_grads, grads):
            success = hyper_map(accu_fn, accumulate_grads, grads)
            return success

        return accumulate_grads_fn

    @staticmethod
    def _get_train_step_fn(network, loss_fn, optimizer, rank_size, scaler, reducer, overflow_still_update=False, ms_jit=False):
        from mindyolo.utils.all_finite import all_finite

        def forward_func(x, label, sizes=None):
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = network(x)
            loss, loss_items = loss_fn(pred, label, x)
            loss *= rank_size
            return scaler.scale(loss), ops.stop_gradient(loss_items)

        grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

        def train_step_func(x, label, sizes=None, optimizer_update=True):
            (loss, loss_items), grads = grad_fn(x, label, sizes)
            grads = reducer(grads)
            unscaled_grads = scaler.unscale(grads)
            grads_finite = all_finite(unscaled_grads)

            if optimizer_update:
                if grads_finite:
                    loss = ops.depend(loss, optimizer(unscaled_grads))
                else:
                    if overflow_still_update:
                        loss = ops.depend(loss, optimizer(unscaled_grads))

            return scaler.unscale(loss), loss_items, unscaled_grads, grads_finite

        @ms.ms_function
        def jit_warpper(*args):
            return train_step_func(*args)

        return train_step_func if not ms_jit else jit_warpper

    def _get_gradreducer(self):
        if self.cfg.is_parallel:
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
            loss_scaler = DynamicLossScaler(scale_value=self.cfg.ms_loss_scaler_value,
                                            scale_factor=self.cfg.scale_factor,
                                            scale_window=self.cfg.scale_window)
        elif ms_loss_scaler == 'static':
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(self.cfg.ms_loss_scaler_value)
        elif ms_loss_scaler in ('none', 'None'):
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(1.0)
        else:
            raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

        return loss_scaler
