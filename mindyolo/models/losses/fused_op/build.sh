nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_intersection_area_kernel.so $(dirname $0)/fused_get_intersection_area_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_ciou_kernel.so $(dirname $0)/fused_get_ciou_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_ciou_diagonal_angle_kernel.so $(dirname $0)/fused_get_ciou_diagonal_angle_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_center_dist_kernel.so $(dirname $0)/fused_get_center_dist_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_boundding_boxes_coord_kernel.so $(dirname $0)/fused_get_boundding_boxes_coord_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_iou_kernel.so $(dirname $0)/fused_get_iou_kernel.cu
nvcc --shared -Xcompiler -fPIC -o $(dirname $0)/fused_get_convex_diagonal_squared_kernel.so $(dirname $0)/fused_get_convex_diagonal_squared_kernel.cu