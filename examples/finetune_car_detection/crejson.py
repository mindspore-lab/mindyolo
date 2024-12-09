import os
import json
from PIL import Image

# 设置数据集路径
dataset_path = "./"
images_path = os.path.join(dataset_path, "images/val")
labels_path = os.path.join(dataset_path, "labels/val")

# 类别映射
categories = [
{"id": 0, "name": "rider"},
{"id": 1, "name": "pedestrian"},
{"id": 2, "name": "trailer"},
{"id": 3, "name": "train"},
{"id": 4, "name": "bus"},
{"id": 5, "name": "car"},
{"id": 6, "name": "truck"},
{"id": 7, "name": "traffic sign"},
{"id": 8, "name": "traffic light"},
{"id": 9, "name": "other person"},
{"id": 10, "name": "motorcycle"},
{"id": 11, "name": "bicycle"},
{"id": 12, "name": "van"},
    # 添加更多类别
]


# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]


# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }


# 处理每个数据集分区
split=''
coco_format = init_coco_format()
annotation_id = 1
imgs_list = os.listdir(os.path.join(images_path, split))
imgs_list.sort()
for img_name in imgs_list:
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(images_path, split, img_name)
        label_path = os.path.join(labels_path, split, img_name.replace("jpg", "txt"))

        img = Image.open(img_path)
        img_width, img_height = img.size
        image_info = {
            "file_name": img_name,
            "id": len(coco_format["images"]) + 1,
            "width": img_width,
            "height": img_height
        }
        coco_format["images"].append(image_info)

        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    category_id, x_center, y_center, width, height = map(float, line.split())
                    bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_info["id"],
                        "category_id": int(category_id) ,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1

    # 为每个分区保存JSON文件
with open(f"./annotations/instances_val2017.json", "w") as json_file:
    json.dump(coco_format, json_file, indent=4)
