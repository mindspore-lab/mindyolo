import os
import ast
import argparse
import shutil
import xml.etree.ElementTree as ET
import collections
import json
from tqdm import tqdm


category_set = ['person']
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_item_id = 0
annotation_id = 0
image_index = 0


def addCatItem(name):
    global category_item_id
    category_item = collections.OrderedDict()
    category_item['supercategory'] = 'none'
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_item_id += 1


def addImgItem(image_id, size):
    file_name = str(image_id).zfill(8) + '.jpg'
    if not size['width']:
        raise Exception('Could not find width tag in xml file.')
    if not size['height']:
        raise Exception('Could not find height tag in xml file.')

    image_item = collections.OrderedDict()

    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    image_item['id'] = image_id
    coco['images'].append(image_item)


def addAnnoItem(image_id, category_id, bbox):
    global annotation_id

    annotation_item = collections.OrderedDict()
    annotation_item['segmentation'] = []

    # segmentation
    seg = []
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['id'] = annotation_id
    annotation_item['ignore'] = 0
    annotation_id += 1
    coco['annotations'].append(annotation_item)


def xxyy2xywhn(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = round(x, 6)
    w = round(w, 6)
    y = round(y, 6)
    h = round(h, 6)
    return x, y, w, h


def xml2txt(xml_path, txt_path):
    in_file = open(xml_path, encoding='utf-8')
    out_file = open(txt_path, 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in category_set or int(difficult) == 1:
            continue
        cls_id = category_set.index(cls)
        xmlbox = obj.find('bndbox')
        x1, x2, y1, y2 = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                          float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

        # clip x2, y2 to normal range
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h

        # xyxy2xywhn
        bbox = (x1, x2, y1, y2)
        bbox = xxyy2xywhn((w, h), bbox)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bbox]) + '\n')


def xml2json(image_index, xml_path):
    bndbox = dict()
    size = dict()
    size['width'] = None
    size['height'] = None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.tag != 'annotation':
        raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

    # elem format: <folder>, <filename>, <size>, <object>
    for elem in root:
        if elem.tag == 'folder' or elem.tag == 'filename' or elem.tag == 'path' or elem.tag == 'source':
            continue

        elif elem.tag == 'size':
            # add image information, like file_name, size, image_id
            for subelem in elem:
                size[subelem.tag] = int(subelem.text)
            addImgItem(image_index, size)

        elif elem.tag == 'object':
            for subelem in elem:
                if subelem.tag == 'name':
                    object_name = subelem.text
                    current_category_id = category_set.index(object_name)

                elif subelem.tag == 'bndbox':
                    for option in subelem:
                        bndbox[option.tag] = int(option.text)

                    bbox = []
                    bbox.append(bndbox['xmin'])
                    bbox.append(bndbox['ymin'])
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])

                    # add bound box information, include area,image_id, bbox, category_id, id and so on
                    addAnnoItem(image_index, current_category_id, bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', type=str, help='annotations directory')
    parser.add_argument('--images_dir', type=str, help='images directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of train set')
    parser.add_argument('--shuffle', type=ast.literal_eval, default=False, help='whether to shuffle the images')
    parser.add_argument('--output_dir', type=str, default='./', help='output yolo-format dataset directory')
    opt = parser.parse_args()

    assert os.path.exists(opt.annotations_dir), f'annotations_dir({opt.annotations_dir}) does not exist'
    assert os.path.exists(opt.images_dir), f'images_dir({opt.images_dir}) does not exist'
    assert opt.train_ratio >= 0.0 and opt.train_ratio <= 1.0

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    image_name_list = os.listdir(opt.images_dir)
    image_num = len(image_name_list)
    assert image_num > 0, 'images_dir is empty'

    print(f'original annotations: {opt.annotations_dir}')
    print(f'original image: {opt.images_dir}')
    print(f'train_ratio: {opt.train_ratio}')
    print(f'output_dir: {opt.output_dir}')

    if not opt.shuffle:
        image_name_list.sort()
    else:
        random.shuffle(image_name_list)

    train_image_num = int(image_num * opt.train_ratio)
    if opt.train_ratio > 0.0:
        train_image_dir = os.path.join(opt.output_dir, 'images', 'train')
        os.makedirs(train_image_dir, exist_ok=True)
        assert not os.listdir(train_image_dir), f'{train_image_dir} already exists'

        train_label_dir = os.path.join(opt.output_dir, 'labels', 'train')
        os.makedirs(train_label_dir, exist_ok=True)
        assert not os.listdir(train_label_dir), f'{train_label_dir} already exists'

        train_txt_origin = open(os.path.join(opt.images_dir, '..', 'train.txt'), 'w')
        train_txt_yolo = open(os.path.join(opt.output_dir, 'train.txt'), 'w')
        for i in tqdm(range(train_image_num), desc='train_set'):
            image_name = image_name_list[i]
            xml_name = os.path.splitext(image_name)[0] + '.xml'
            new_image_name = str(i) + os.path.splitext(image_name)[1]

            shutil.copy2(os.path.join(opt.images_dir, image_name), os.path.join(train_image_dir, new_image_name))

            train_txt_origin.write(image_name + '\n')

            train_txt_yolo.write('./images/train/' + new_image_name + '\n')

            xml_path = os.path.join(opt.annotations_dir, xml_name)
            txt_path = os.path.join(train_label_dir, str(i) + '.txt')
            xml2txt(xml_path, txt_path)

        train_txt_origin.close()
        train_txt_yolo.close()

    if opt.train_ratio < 1.0:
        val_image_dir = os.path.join(opt.output_dir, 'images', 'val')
        os.makedirs(val_image_dir, exist_ok=True)
        assert not os.listdir(val_image_dir), f'{val_image_dir} already exists'

        val_label_dir = os.path.join(opt.output_dir, 'labels', 'val')
        os.makedirs(val_label_dir, exist_ok=True)
        assert not os.listdir(val_label_dir), f'{val_label_dir} already exists'

        val_annotation_dir = os.path.join(opt.output_dir, 'annotations')
        os.makedirs(val_annotation_dir, exist_ok=True)
        assert not os.listdir(val_annotation_dir), f'{val_annotation_dir} already exists'

        val_txt_origin = open(os.path.join(opt.images_dir, '..', 'val.txt'), 'w')
        val_txt_yolo = open(os.path.join(opt.output_dir, 'val.txt'), 'w')
        for i in tqdm(range(train_image_num, image_num), desc='val_set'):
            image_name = image_name_list[i]
            xml_name = os.path.splitext(image_name)[0] + '.xml'
            new_image_name = str(i) + os.path.splitext(image_name)[1]

            shutil.copy2(os.path.join(opt.images_dir, image_name), os.path.join(val_image_dir, new_image_name))

            val_txt_origin.write(image_name + '\n')

            val_txt_yolo.write('./images/val/' + new_image_name + '\n')

            xml_path = os.path.join(opt.annotations_dir, xml_name)
            txt_path = os.path.join(val_label_dir, str(i) + '.txt')
            xml2txt(xml_path, txt_path)

            xml2json(i, xml_path)

        for categoryname in category_set:
            addCatItem(categoryname)

        val_txt_origin.close()
        val_txt_yolo.close()

        # save ground truth json file
        json_file = os.path.join(opt.output_dir, 'annotations', 'instances_val2017.json')
        json.dump(coco, open(json_file, 'w'))

    print(f'dataset convert compeleted, yolo-format dataset saved in {opt.output_dir}')
