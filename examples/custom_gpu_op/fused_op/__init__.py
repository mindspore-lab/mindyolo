import os

from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.ops import DataType, CustomRegOp


fused_ops_dir = os.path.dirname(__file__)

fused_get_ciou_op_path = fused_ops_dir + "/fused_get_ciou_kernel.so" + ":FusedGetCiou"
fused_get_ciou_op_bprop_path = fused_ops_dir + "/fused_get_ciou_kernel.so" + ":FusedGetCiouBprop"
fused_get_center_dist_op_path = fused_ops_dir + "/fused_get_center_dist_kernel.so" + ":FusedGetCenterDist"
fused_get_center_dist_op_bprop_path = fused_ops_dir + "/fused_get_center_dist_kernel.so" + ":FusedGetCenterDistBprop"
fused_get_convex_diagonal_squared_path = fused_ops_dir + "/fused_get_convex_diagonal_squared_kernel.so" + ":FusedGetConvexDiagonalSquared"
fused_get_convex_diagonal_squared_grad_path = fused_ops_dir + "/fused_get_convex_diagonal_squared_kernel.so" + ":FusedGetConvexDiagonalSquaredGrad"
fused_get_iou_op_path = fused_ops_dir + "/fused_get_iou_kernel.so" + ":FusedGetIou"
fused_get_iou_op_bprop_path = fused_ops_dir + "/fused_get_iou_kernel.so" + ":FusedGetIouBprop"
fused_get_ciou_diagonal_angle_path = fused_ops_dir + "/fused_get_ciou_diagonal_angle_kernel.so" + ":FusedGetCiouDiagonalAngle"
fused_get_ciou_diagonal_angle_grad_path = fused_ops_dir + "/fused_get_ciou_diagonal_angle_kernel.so" + ":FusedGetCiouDiagonalAngleGrad"
fused_get_boundding_boxes_coord_path = fused_ops_dir + "/fused_get_boundding_boxes_coord_kernel.so" + ":FusedGetBounddingBoxesCoord"
fused_get_boundding_boxes_coord_grad_path = fused_ops_dir+"/fused_get_boundding_boxes_coord_kernel.so" + ":FusedGetBounddingBoxesCoordGrad"
fused_get_intersection_area_path = fused_ops_dir + "/fused_get_intersection_area_kernel.so" + ":FusedGetIntersectionArea"
fused_get_intersection_area_grad_path = fused_ops_dir + "/fused_get_intersection_area_kernel.so" + ":FusedGetIntersectionAreaGrad"

fused_get_ciou_gpu_info = CustomRegOp() \
    .input(0, "v") \
    .input(1, "iou") \
    .input(2, "rho2") \
    .input(3, "c2") \
    .output(0, "alpha") \
    .output(1, "out") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_ciou_bprop_gpu_info = CustomRegOp() \
    .input(0, "v") \
    .input(1, "iou") \
    .input(2, "rho2") \
    .input(3, "c2") \
    .input(4, "d_alpha") \
    .input(5, "d_out") \
    .output(0, "d_v") \
    .output(1, "d_iou") \
    .output(2, "d_rho2") \
    .output(3, "d_c2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_center_dist_gpu_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b1_y1") \
    .input(3, "b1_y2") \
    .input(4, "b2_x1") \
    .input(5, "b2_x2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .output(0, "out") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_center_dist_bprop_gpu_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b1_y1") \
    .input(3, "b1_y2") \
    .input(4, "b2_x1") \
    .input(5, "b2_x2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .input(8, "d_out") \
    .output(0, "d_b1_x1") \
    .output(1, "d_b1_x2") \
    .output(2, "d_b1_y1") \
    .output(3, "d_b1_y2") \
    .output(4, "d_b2_x1") \
    .output(5, "d_b2_x2") \
    .output(6, "d_b2_y1") \
    .output(7, "d_b2_y2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_convex_diagonal_squared_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b2_x1") \
    .input(3, "b2_x2") \
    .input(4, "b1_y1") \
    .input(5, "b1_y2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .output(0, "out") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_iou_gpu_info = CustomRegOp() \
    .input(0, "w1") \
    .input(1, "h1") \
    .input(2, "w2") \
    .input(3, "h2") \
    .input(4, "inter") \
    .output(0, "out") \
    .output(1, "val_union") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_convex_diagonal_squared_grad_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b2_x1") \
    .input(3, "b2_x2") \
    .input(4, "b1_y1") \
    .input(5, "b1_y2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .input(8, "dout") \
    .output(0, "d_b1_x1") \
    .output(1, "d_b1_x2") \
    .output(2, "d_b2_x1") \
    .output(3, "d_b2_x2") \
    .output(4, "d_b1_y1") \
    .output(5, "d_b1_y2") \
    .output(6, "d_b2_y1") \
    .output(7, "d_b2_y2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_iou_bprop_gpu_info = CustomRegOp() \
    .input(0, "w1") \
    .input(1, "h1") \
    .input(2, "w2") \
    .input(3, "h2") \
    .input(4, "inter") \
    .input(5, "d_out") \
    .input(6, "d_val_union") \
    .output(0, "d_w1") \
    .output(1, "d_h1") \
    .output(2, "d_w2") \
    .output(3, "d_h2") \
    .output(4, "d_inter") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_ciou_diagonal_angle_info = CustomRegOp() \
    .input(0, "w1") \
    .input(1, "h1") \
    .input(2, "w2") \
    .input(3, "h2") \
    .output(0, "out") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_ciou_diagonal_angle_grad_info = CustomRegOp() \
    .input(0, "w1") \
    .input(1, "h1") \
    .input(2, "w2") \
    .input(3, "h2") \
    .input(4, "out") \
    .output(0, "w1_diff") \
    .output(1, "h1_diff") \
    .output(2, "w2_diff") \
    .output(3, "h2_diff") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_boundding_boxes_coord_gpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "y1") \
    .input(2, "w1") \
    .input(3, "h1") \
    .input(4, "x2") \
    .input(5, "y2") \
    .input(6, "w2") \
    .input(7, "h2") \
    .output(0, "b1_x1") \
    .output(1, "b1_y1") \
    .output(2, "b1_x2") \
    .output(3, "b1_y2") \
    .output(4, "b2_x1") \
    .output(5, "b2_y1") \
    .output(6, "b2_x2") \
    .output(7, "b2_y2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_boundding_boxes_coord_bprop_gpu_info = CustomRegOp() \
    .input(0, "d_b1_x1") \
    .input(1, "d_b1_x2") \
    .input(2, "d_b1_y1") \
    .input(3, "d_b1_y2") \
    .input(4, "d_b2_x1") \
    .input(5, "d_b2_x2") \
    .input(6, "d_b2_y1") \
    .input(7, "d_b2_y2") \
    .output(0, "d_x1") \
    .output(1, "d_y1") \
    .output(2, "d_w1") \
    .output(3, "d_h1") \
    .output(4, "d_x2") \
    .output(5, "d_y2") \
    .output(6, "d_w2") \
    .output(7, "d_h2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_intersection_area_gpu_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b2_x1") \
    .input(3, "b2_x2") \
    .input(4, "b1_y1") \
    .input(5, "b1_y2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .output(0, "inter") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

fused_get_intersection_area_gpu_grad_info = CustomRegOp() \
    .input(0, "b1_x1") \
    .input(1, "b1_x2") \
    .input(2, "b2_x1") \
    .input(3, "b2_x2") \
    .input(4, "b1_y1") \
    .input(5, "b1_y2") \
    .input(6, "b2_y1") \
    .input(7, "b2_y2") \
    .input(8, "d_inter") \
    .output(0, "d_b1_x1") \
    .output(1, "d_b1_x2") \
    .output(2, "d_b2_x1") \
    .output(3, "d_b2_x2") \
    .output(4, "d_b1_y1") \
    .output(5, "d_b1_y2") \
    .output(6, "d_b2_y1") \
    .output(7, "d_b2_y2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()


def get_ciou_bprop(v, iou, rho2, c2, out, dout):
    fused_get_ciou_bprop = ops.Custom(fused_get_ciou_op_bprop_path,
                                      out_shape=(v.shape, iou.shape, rho2.shape, c2.shape),
                                      out_dtype=(mstype.float32, mstype.float32, mstype.float32, mstype.float32),
                                      func_type="aot", reg_info=fused_get_ciou_bprop_gpu_info)
    res = fused_get_ciou_bprop(v, iou, rho2, c2, dout[0], dout[1])
    return res


fused_get_ciou = ops.Custom(fused_get_ciou_op_path,
                            out_shape=lambda v, iou, rho2, c2: (v, v),
                            out_dtype=(mstype.float32, mstype.float32),
                            func_type="aot", bprop=get_ciou_bprop, reg_info=fused_get_ciou_gpu_info)


def get_center_dist_bprop(b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, out, dout):
    fused_get_center_dist_bprop = ops.Custom(fused_get_center_dist_op_bprop_path,
                                             out_shape=(b1_x1.shape, b1_x2.shape, b1_y1.shape, b1_y2.shape,
                                                        b2_x1.shape, b2_x2.shape, b2_y1.shape, b2_y2.shape),
                                             out_dtype=(mstype.float32, mstype.float32, mstype.float32, mstype.float32,
                                                        mstype.float32, mstype.float32, mstype.float32, mstype.float32),
                                             func_type="aot", reg_info=fused_get_center_dist_bprop_gpu_info)
    res = fused_get_center_dist_bprop(b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, dout)
    return res


fused_get_center_dist = ops.Custom(fused_get_center_dist_op_path,
                                   out_shape=lambda b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2: (b1_x1),
                                   out_dtype=(mstype.float32),
                                   func_type="aot", bprop=get_center_dist_bprop, reg_info=fused_get_center_dist_gpu_info)


def fused_get_convex_diagonal_squared_bprop(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, out, dout):
    out_shape = (b1_x1.shape, b1_x2.shape, b2_x1.shape, b2_x2.shape, b1_y1.shape, b1_y2.shape, b2_y1.shape, b2_y2.shape)
    out_dtype = (b1_x1.dtype, b1_x2.dtype, b2_x1.dtype, b2_x2.dtype, b1_y1.dtype, b1_y2.dtype, b2_y1.dtype, b2_y2.dtype)
    op = ops.Custom(fused_get_convex_diagonal_squared_grad_path,
                    out_shape=out_shape,
                    out_dtype=out_dtype,
                    reg_info=fused_get_convex_diagonal_squared_grad_info,
                    func_type="aot")
    return op(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, dout)


fused_get_convex_diagonal_squared = ops.Custom(
    fused_get_convex_diagonal_squared_path,
    out_shape=lambda b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2: b1_x1,
    out_dtype=lambda b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2: b1_x1,
    bprop=fused_get_convex_diagonal_squared_bprop,
    reg_info=fused_get_convex_diagonal_squared_info,
    func_type="aot")


def fused_get_ciou_diagonal_angle_bprop(w1, h1, w2, h2, out, dout):
    out_shape = (w1.shape, h1.shape, w2.shape, h2.shape)
    out_dtype = (w1.dtype, h1.dtype, w2.dtype, h2.dtype)
    op = ops.Custom(fused_get_ciou_diagonal_angle_grad_path,
                    out_shape=out_shape,
                    out_dtype=out_dtype,
                    reg_info=fused_get_ciou_diagonal_angle_grad_info,
                    func_type="aot")
    return op(w1, h1, w2, h2, dout)


fused_get_ciou_diagonal_angle = ops.Custom(
    fused_get_ciou_diagonal_angle_path,
    out_shape=lambda w1, h1, w2, h2: w1,
    out_dtype=lambda w1, h1, w2, h2: w1,
    bprop=fused_get_ciou_diagonal_angle_bprop,
    reg_info=fused_get_ciou_diagonal_angle_info,
    func_type="aot")


def fused_get_iou_bprop(w1, h1, w2, h2, inter, out, dout):
    fused_get_iou_bprop = ops.Custom(fused_get_iou_op_bprop_path,
                                    out_shape=(w1.shape, h1.shape, w2.shape, h2.shape, inter.shape),
                                    out_dtype=(
                                        mstype.float32, mstype.float32, mstype.float32, mstype.float32, mstype.float32),
                                    func_type="aot", reg_info=fused_get_iou_bprop_gpu_info)
    res = fused_get_iou_bprop(w1, h1, w2, h2, inter, dout[0], dout[1])
    return res


fused_get_iou = ops.Custom(fused_get_iou_op_path,
                          out_shape=lambda w1, h1, w2, h2, inter: (w1, w1),
                          out_dtype=(mstype.float32, mstype.float32),
                          func_type="aot", bprop=fused_get_iou_bprop, reg_info=fused_get_iou_gpu_info)


def fused_get_boundding_boxes_coord_bprop(x1, y1, w1, h1, x2, y2, w2, h2, out, dout):
    out_shape = (x1.shape, y1.shape, w1.shape, h1.shape,
                 x2.shape, y2.shape, w2.shape, h2.shape)
    out_dtype = (mstype.float32, mstype.float32, mstype.float32, mstype.float32,
                 mstype.float32, mstype.float32, mstype.float32, mstype.float32)
    op = ops.Custom(fused_get_boundding_boxes_coord_grad_path, out_shape=out_shape, out_dtype=out_dtype,
                    func_type='aot', reg_info=fused_get_boundding_boxes_coord_bprop_gpu_info)
    return op(dout[0], dout[1], dout[2], dout[3], dout[4], dout[5], dout[6], dout[7])


fused_get_boundding_boxes_coord = ops.Custom(fused_get_boundding_boxes_coord_path,out_shape=lambda x1, y1, w1, h1, x2, y2, w2, h2: (
    x1, y1, w1, h1, x2, y2, w2, h2),
    out_dtype=lambda x1, y1, w1, h1, x2, y2, w2, h2: (
    x1, y1, w1, h1, x2, y2, w2, h2),
    func_type='aot', bprop=fused_get_boundding_boxes_coord_bprop, reg_info=fused_get_boundding_boxes_coord_gpu_info)


def fused_get_intersection_area_bprop(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, out, dout):
    out_shape = (b1_x1.shape, b1_x2.shape, b2_x1.shape, b2_x2.shape,
                 b1_y1.shape, b1_y2.shape, b2_y1.shape, b2_y2.shape)
    out_dtype = (mstype.float32, mstype.float32, mstype.float32, mstype.float32,
                 mstype.float32, mstype.float32, mstype.float32, mstype.float32)
    op=ops.Custom(fused_get_intersection_area_grad_path, out_shape=out_shape, out_dtype=out_dtype,
                    func_type='aot', reg_info=fused_get_intersection_area_gpu_grad_info)
    
    return op(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, dout)


fused_get_intersection_area = ops.Custom(
    fused_get_intersection_area_path,     
    out_shape=lambda b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2: b1_x1,
    out_dtype=lambda b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2: b1_x1,
    func_type='aot', bprop=fused_get_intersection_area_bprop, reg_info=fused_get_intersection_area_gpu_info)
