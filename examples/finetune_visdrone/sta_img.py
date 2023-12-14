# -*- coding: utf-8 -*-
'''
@File Description:
# json文件images信息
'''

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def js_img_sta(js_path, csv_path, png_shape_path, png_shapeRate_path):
    print('json read...\n')
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)

    df_img = pd.DataFrame(data['images'])

    if png_shape_path is not None:
        sns.jointplot('height', 'width', data=df_img, kind='hex')
        plt.savefig(png_shape_path)
        plt.close()
        print('png save to', png_shape_path)
    if png_shapeRate_path is not None:
        df_img['shape_rate'] = (df_img['width'] / df_img['height']).round(1)
        df_img['shape_rate'].value_counts().sort_index().plot(kind='bar', title='images shape rate')
        plt.savefig(png_shapeRate_path)
        plt.close()
        print('png save to', png_shapeRate_path)

    if csv_path is not None:
        df_img.to_csv(csv_path)
        print('csv save to', csv_path)


def get_args():
    parser = argparse.ArgumentParser(description='Json Images Infomation Statistic')

    # parameters
    parser.add_argument('--json_path', type=str,
                        help='json path to statistic images information')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='csv path to save statistic images information, default None, do not save')
    parser.add_argument('--png_shape_path', type=str, default=None,
                        help='png path to save statistic images shape information, default None, do not save')
    parser.add_argument('--png_shapeRate_path', type=str, default=None,
                        help='png path to save statistic images shape rate information, default None, do not save')
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100, '-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args


if __name__ == '__main__':
    args = get_args()
    js_img_sta(args.json_path, args.csv_path, args.png_shape_path, args.png_shapeRate_path)

