import os
import json
import numpy as np
from PIL import Image

dataset_path = "./"
images_path = os.path.join(dataset_path, "valid/images")
labels_path = os.path.join(dataset_path, "valid/labels")

categories = [
{"id": 0, "name": "back_bumper"},
{"id": 1, "name": "back_door"},
{"id": 2, "name": "back_glass"},
{"id": 3, "name": "back_left_door"},
{"id": 4, "name": "back_left_light"},
{"id": 5, "name": "back_light"},
{"id": 6, "name": "back_right_door"},
{"id": 7, "name": "back_right_light"},
{"id": 8, "name": "front_bumper"},
{"id": 9, "name": "front_door"},
{"id": 10, "name": "front_glass"},
{"id": 11, "name": "front_left_door"},
{"id": 12, "name": "front_left_light"},
{"id": 13, "name": "front_light"},
{"id": 14, "name": "front_right_door"},
{"id": 15, "name": "front_right_light"},
{"id": 16, "name": "hood"},
{"id": 17, "name": "left_mirror"},
{"id": 18, "name": "object"},
{"id": 19, "name": "right_mirror"},
{"id": 20, "name": "tailgate"},
{"id": 21, "name": "trunk"},
{"id": 22, "name": "wheel"},
]


# init_coco_format
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (xy1, xy2, ...) to (xywh)
    boxes = []
    x, y = segments.T  # segment xy
    boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes))  # xywh

# make sure images list is in order
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
                    values = line.strip().split()
                    category_id = int(values[0])
                    segments = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
                    bbox = np.squeeze(segments2boxes(segments)).tolist()
                    bbox = [i * img_width for i in bbox]

                    segmentation = np.array(values[1:], dtype=np.float32)
                    segmentation = np.expand_dims([i * img_width for i in segmentation], axis=0).tolist()
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_info["id"],
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1

if not os.path.exists("./annotations"):
    os.mkdir("./annotations")

# create instances_val2017.json file
with open(f"./annotations/instances_val2017.json", "w") as json_file:
    json.dump(coco_format, json_file, indent=4)