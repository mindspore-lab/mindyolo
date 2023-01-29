import argparse
import ast

def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description='Train', parents=[parents] if parents else [])
    parser.add_argument('--ms_strategy', type=str, default='StaticShape', help='train strategy, StaticCell/StaticShape/MultiShape/DynamicShape')
    parser.add_argument('--ms_mode', type=int, default=0, help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--ms_loss_scaler', type=str, default='static', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=1024.0, help='static loss scale value')
    parser.add_argument('--ms_optim_loss_scale', type=float, default=1.0, help='optimizer loss scale')
    parser.add_argument('--ms_grad_sens', type=float, default=1024.0, help='gard sens')
    parser.add_argument('--num_parallel_workers', type=int, default=4, help='num parallel worker for dataloader')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=True, help='overflow still update')
    parser.add_argument('--clip_grad', type=ast.literal_eval, default=False, help='clip grad')
    parser.add_argument('--ema', type=ast.literal_eval, default=True, help='ema')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help='Distribute train or not')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--weight', type=str, default='', help='initial weight path')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weight path')
    parser.add_argument('--epochs', type=int, default=300, help="total train epochs")
    parser.add_argument('--total_batch_size', type=int, default=32, help='total batch size for all device')
    parser.add_argument('--accumulate', type=int, default=1, help='grad accumulate step, recommended when batch-size is less than 64')
    parser.add_argument('--img_size', type=list, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--run_eval', type=ast.literal_eval, default=False, help='run eval')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')

    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--nms_time_limit', type=float, default=10.0, help='time limit for NMS')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False, help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, default='momentum', help='select optimizer')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False, help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=False, help='existing project/name ok, do not increment')
    parser.add_argument('--linear_lr', type=ast.literal_eval, default=False, help='linear LR')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--freeze', type=list, default=[],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5_metric', type=ast.literal_eval, default=False, help='assume maximum recall as 1.0 in AP calculation')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/', help='ModelArts: obs path to dataset folder')
    return parser


def get_args_test(parents=None):
    parser = argparse.ArgumentParser(description='Test', parents=[parents] if parents else [])
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--nms_time_limit', type=float, default=20.0, help='time limit for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False, help='train multi-class data as single-class')
    parser.add_argument('--augment', type=ast.literal_eval, default=False, help='augmented inference')
    parser.add_argument('--verbose', type=ast.literal_eval, default=False, help='report mAP by class')
    parser.add_argument('--save_txt', type=ast.literal_eval, default=False, help='save results to *.txt')
    parser.add_argument('--save_hybrid', type=ast.literal_eval, default=False, help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', type=ast.literal_eval, default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', type=ast.literal_eval, default=False, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='./run_test', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=False, help='existing project/name ok, do not increment')
    parser.add_argument('--no_trace', type=ast.literal_eval, default=False, help='don`t trace model')
    parser.add_argument('--v5_metric', type=ast.literal_eval, default=False, help='assume maximum recall as 1.0 in AP calculation')
    return parser

def get_args_310(parents=None):
    parser = argparse.ArgumentParser(description='Export', parents=[parents] if parents else [])

    # export
    parser.add_argument('--ms_mode', type=int, default=0, help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weight', type=str, default='yolov7_300.ckpt', help='model.ckpt path')
    parser.add_argument('--per_batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--file_format', type=str, default='MINDIR', help='treat as single-class dataset')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False, help='train multi-class data as single-class')

    # preprocess
    parser.add_argument('--output_path', type=str, default='./', help='output preprocess data path')

    # postprocess
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--result_path', type=str, default='./result_Files', help='path to 310 infer result floder')
    parser.add_argument('--project', default='./run_test', help='save to project/name')

    return parser
