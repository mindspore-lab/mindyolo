import os
import cv2
from tqdm import tqdm
import json
import argparse
from skimage import io
import pandas as pd

categories_info = [
    {"id": 0, "name": "ignored regions"},
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
    {"id": 11, "name": "others"}
]


def data_convert(img_dir_src, anno_dir_src, json_path, img_dir_dst, anno_dir_dst, txt_path):
    if not os.path.exists(img_dir_dst):
        os.makedirs(img_dir_dst)
    if not os.path.exists(anno_dir_dst):
        os.makedirs(anno_dir_dst)
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))
    file_list = os.listdir(img_dir_src)
    box_num = 0
    anno_infos = []
    img_infos = []
    df_img = pd.DataFrame(None, columns=['img_name'])
    for idx, img_name_src in enumerate(tqdm(file_list)):
        anno_name_src = img_name_src.replace('.jpg', '.txt')
        img_name_dst = '%06d.jpg' % idx
        anno_name_dst = img_name_dst.replace('.jpg', '.txt')
        img_path_src = os.path.join(img_dir_src, img_name_src)
        anno_path_src = os.path.join(anno_dir_src, anno_name_src)
        img_path_dst = os.path.join(img_dir_dst, img_name_dst)
        anno_path_dst = os.path.join(anno_dir_dst, anno_name_dst)
        df_img.loc[len(df_img)] = img_path_dst
        img = io.imread(img_path_src)
        io.imsave(img_path_dst, img)
        img_height, img_width = img.shape[:2]

        img_info = {}
        img_info["file_name"] = img_name_dst
        img_info["height"] = img_height
        img_info["width"] = img_width
        img_info["id"] = idx
        img_infos.append(img_info)

        df_anno = pd.read_csv(anno_path_src, index_col=None, header=None,
                              names=['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'category',
                                     'truncation', 'occlusion'])
        df_anno['bbox_center_x'] = (df_anno['bbox_left'] + 0.5 * df_anno['bbox_width']) / img_width
        df_anno['bbox_center_y'] = (df_anno['bbox_top'] + 0.5 * df_anno['bbox_height']) / img_height
        df_anno['bbox_w'] = df_anno['bbox_width'] / img_width
        df_anno['bbox_h'] = df_anno['bbox_height'] / img_height

        df_anno_dst = df_anno[['category', 'bbox_center_x', 'bbox_center_y', 'bbox_w', 'bbox_h']]
        df_anno_dst = df_anno_dst.round({"bbox_center_x": 6, "bbox_center_y": 6, "bbox_w": 6, "bbox_h": 6})
        df_anno_dst.to_csv(anno_path_dst, header=None, index=None, sep=' ')

        for i, row in df_anno.iterrows():
            anno_info = {}
            bbox_xywh = [int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])]
            anno_info["image_id"] = idx
            anno_info["score"] = float(row['score'])
            anno_info["bbox"] = bbox_xywh
            anno_info["category_id"] = int(row['category'])
            anno_info["id"] = box_num
            anno_info["iscrowd"] = 0
            anno_info["segmentation"] = []
            anno_info["area"] = bbox_xywh[2] * bbox_xywh[3]
            anno_info["truncation"] = float(row['truncation'])
            anno_info["occlusion"] = float(row['occlusion'])
            box_num += 1

            anno_infos.append(anno_info)

    df_img.to_csv(txt_path, header=None, index=None)
    data_info = {}
    data_info["images"] = img_infos
    data_info["annotations"] = anno_infos
    data_info["categories"] = categories_info
    # print(data_info)
    json_str = json.dumps(data_info)

    with open(json_path, 'w') as json_file:
        json_file.write(json_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='', type=str, help='img_dir')
    parser.add_argument('--gt_dir', default='', type=str, help='gt_dir')
    parser.add_argument('--json_path', default='', type=str, help='gt_dir')
    parser.add_argument('--img_dir_dst', default='', type=str, help='img_dir_dst')
    parser.add_argument('--gt_dir_dst', default='', type=str, help='gt_dir_dst')
    parser.add_argument('--txt_path', default='', type=str, help='txt_path')
    opt = parser.parse_args()
    data_convert(opt.img_dir, opt.gt_dir, opt.json_path, opt.img_dir_dst, opt.gt_dir_dst, opt.txt_path)
