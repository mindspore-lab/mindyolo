import os
import json

# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }

# 添加图像信息
def add_image(coco, image_id, file_name, height, width):
    coco["images"].append({
        "id": image_id,
        "file_name": file_name,
        "height": height,
        "width": width
    })

# 添加类别信息
def add_category(coco, category_id, category_name):
    coco["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": "none"
    })

# 添加标注信息
def add_annotation(coco, annotation_id, image_id, category_id, bbox, area, iscrowd):
    coco["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": iscrowd
    })

# 主转换函数
def convert_yolo_to_coco(yolo_label_dir, coco_json_path, img_dir, categories):
    coco = init_coco_format()
    annotation_id = 1
    for image_id, file_name in enumerate(os.listdir(img_dir)):
        file_path = os.path.join(yolo_label_dir, file_name.replace(".jpg", ".txt"))
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            lines = f.readlines()
            height, width = (540, 960)  # 从图像获取真实尺寸
            add_image(coco, image_id+1, file_name, height, width)
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                w = w * width
                h = h * height
                bbox = [x_min, y_min, w, h]
                area = w * h
                add_annotation(coco, annotation_id, image_id+1, int(class_id), bbox, area, 0)
                annotation_id += 1
    for i, category in enumerate(categories):
        add_category(coco, i + 1, category)

    with open(coco_json_path, 'w') as f:
        json.dump(coco, f)

# 使用
yolo_label_dir = './labels/train'
img_dir = './images/train'
coco_json_path = './annotations/train.json'
categories = ['car',"bus","van","others"]  # 你的类别列表
convert_yolo_to_coco(yolo_label_dir, coco_json_path, img_dir, categories)