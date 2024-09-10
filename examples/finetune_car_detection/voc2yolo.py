import xml.etree.ElementTree as ET
import sys
import os.path


class XmlParse:
    def __init__(self, file_path):
        # 初始化成员变量：self.tree 和 self.root 分别用于存储XML文件解析后的ElementTree对象和根节点；self.xml_file_path 存储传入的XML文件路径。
        self.tree = None
        self.root = None
        self.xml_file_path = file_path

    # 使用 try...except...finally 结构处理可能出现的异常情况。
    def ReadXml(self):  # 该方法用于读取XML文件并解析为ElementTree对象。
        try:
            self.tree = ET.parse(self.xml_file_path)  # 使用 xml.etree.ElementTree.parse() 方法解析XML文件，将结果赋值给 self.tree
            self.root = self.tree.getroot()  # 获取XML文件的根节点并赋值给 self.root。
        except Exception as e:  # 在 except Exception as e 块内，捕获并打印解析失败的错误信息，并通过 sys.exit() 终止程序执行。
            print("parse xml faild!")
            sys.exit()
        else:
            pass
        finally:  # finally 块会在不论是否出现异常的情况下都会被执行，这里返回解析好的 self.tree。
            return self.tree

    def WriteXml(self, destfile):
        dses_xml_file = os.path.abspath(destfile)
        self.tree.write(dses_xml_file, encoding="utf-8", xml_declaration=True)


def xml2txt(xml, labels, name_list, img_path):
    for i, j in zip(os.listdir(xml), os.listdir(img_path)):
        p = os.path.join(xml + '/' + i)  # xml路径
        xml_file = os.path.abspath(p)  # 绝对路径
        parse = XmlParse(xml_file)
        tree = parse.ReadXml()  # xml树
        root = tree.getroot()  # 根节点

        W = float(root.find('size').find('width').text)
        H = float(root.find('size').find('height').text)

        fil_name = root.find('filename').text[:-4]
        if not os.path.exists(labels):  # 如果路径不存在则创建
            os.mkdir(labels)
        out = open(labels + '/' + fil_name + '.txt', 'w+')
        for obj in root.iter('object'):

            x_min = float(obj.find('bndbox').find('xmin').text)
            x_max = float(obj.find('bndbox').find('xmax').text)
            y_min = float(obj.find('bndbox').find('ymin').text)
            y_max = float(obj.find('bndbox').find('ymax').text)
            # print(f'------------------------{i}-----------------------')
            # print('W:', W, 'H:', H)
            # 计算公式
            xcenter = x_min + (x_max - x_min) / 2
            ycenter = y_min + (y_max - y_min) / 2
            w = x_max - x_min
            h = y_max - y_min
            # 目标框的中心点 宽高
            # print('center_X: ', xcenter)
            # print('center_Y: ', ycenter)
            # print('target box_w: ', w)
            # print('target box_h: ', h)
            # 归一化
            xcenter = round(xcenter / W, 6)
            ycenter = round(ycenter / H, 6)
            w = round(w / W, 6)
            h = round(h / H, 6)
            #
            # print('>>>>>>>>>>')
            # print(xcenter)
            # print(ycenter)
            # print(w)
            # print(h)

            class_dict = {name: i for i, name in enumerate(name_list)}
            class_name = obj.find('name').text
            if class_name not in name_list:
                pass
            else:
                class_id = class_dict[class_name]
                # print('类别: ', class_id)
                # print("创建成功: {}".format(fil_name + '.txt'))
                # print('----------------------------------------------------')
                out.write(str(class_id) + " " + str(xcenter) + " " + str(ycenter) + " " + str(w) + " " + str(h) + "\n")

                # show_img
                # m = os.path.join(img_path + '/' + j)
                # block = cv2.imread(m)
                # cv2.rectangle(block, pt1=(int((xcenter - w / 2) * W), int((ycenter - h / 2) * H)),
                #               pt2=(int((xcenter + w / 2) * W), int((ycenter + h / 2) * H)),
                #               color=(0, 255, 0), thickness=2)
                # cv2.imshow('block', block)
                # cv2.waitKey(300)


def folder_Path():
    img_path = './images/val'
    xml_path = './labels_xml/val'  # xml路径
    labels = './val'  # 转txt路径
    name_list = ['rider', 'pedestrian', 'trailer', 'train', 'bus', 'car', 'truck', 'traffic sign', 'traffic light', 'other person', 'motorcycle', 'bicycle', "van"]  # 类别名

    xml2txt(xml_path, labels, name_list, img_path)


if __name__ == '__main__':
    folder_Path()
  