import os
from pathlib import Path
import argparse
import shutil
import xml.etree.ElementTree as ET
import collections
import json
from tqdm import tqdm


category_set = ['person', 'hat']
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
    parser.add_argument('--root_dir', default='', type=str, help='root directory of data set')
    opt = parser.parse_args()

    # generate directory structure
    root_dir = opt.root_dir
    new_dir = os.path.join(root_dir, '..', 'SHWD')
    os.makedirs(os.path.join(new_dir, 'images', 'train'))
    os.makedirs(os.path.join(new_dir, 'images', 'val'))
    os.makedirs(os.path.join(new_dir, 'labels', 'train'))
    os.makedirs(os.path.join(new_dir, 'labels', 'val'))
    os.makedirs(os.path.join(new_dir, 'annotations'))

    train_txt_yolo = open(os.path.join(new_dir, 'train.txt'), 'w')
    val_txt_yolo = open(os.path.join(new_dir, 'val.txt'), 'w')

    images_path = os.path.join(root_dir, 'JPEGImages')
    labels_path = os.path.join(root_dir, 'Annotations')

    train_set_txt = os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt')
    with open(train_set_txt, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='train_set'):
            stem = line.strip('\n')
            old_path = os.path.join(images_path, stem + '.jpg')
            if not os.path.exists(old_path):
                old_path = os.path.join(images_path, stem + '.JPG')

            # copy train_set image to new path
            new_images_path = os.path.join(new_dir, 'images', 'train')
            shutil.copy(old_path, new_images_path)

            # rename image_file to continuous number
            old_name = Path(old_path).name
            new_stem = str(image_index).zfill(8)
            os.rename(os.path.join(new_images_path, old_name), os.path.join(new_images_path, new_stem + '.jpg'))

            # write the relative path of image to train.txt
            train_txt_yolo.write('./images/train/' + new_stem + '.jpg' + '\n')

            # convert xml file to txt file
            xml_path = os.path.join(labels_path, stem + '.xml')
            txt_path = os.path.join(new_dir, 'labels', 'train', new_stem + '.txt')
            xml2txt(xml_path, txt_path)

            image_index += 1

    val_set_txt = os.path.join(root_dir, 'ImageSets', 'Main', 'test.txt')
    with open(val_set_txt, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='val_set'):
            stem = line.strip('\n')
            old_path = os.path.join(images_path, stem + '.jpg')
            if not os.path.exists(old_path):
                old_path = os.path.join(images_path, stem + '.JPG')

            # copy val_set image to new path
            new_images_path = os.path.join(new_dir, 'images', 'val')
            shutil.copy(old_path, new_images_path)

            # rename image_file to continuous number
            old_name = Path(old_path).name
            new_stem = str(image_index).zfill(8)
            os.rename(os.path.join(new_images_path, old_name), os.path.join(new_images_path, new_stem + '.jpg'))

            # write the relative path of image to val.txt
            val_txt_yolo.write('./images/val/' + new_stem + '.jpg' + '\n')

            # convert xml file to txt file
            xml_path = os.path.join(labels_path, stem + '.xml')
            txt_path = os.path.join(new_dir, 'labels', 'val', new_stem + '.txt')
            xml2txt(xml_path, txt_path)

            # convert xml file to json file
            xml2json(image_index, xml_path)

            image_index += 1

    for categoryname in category_set:
        addCatItem(categoryname)

    train_txt_yolo.close()
    val_txt_yolo.close()

    # save ground truth json file
    json_file = os.path.join(new_dir, 'annotations', 'instances_val2017.json')
    json.dump(coco, open(json_file, 'w'))
