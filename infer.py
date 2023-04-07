import os

from mindyolo.engine.enginer import Enginer
from mindyolo.utils.config import parse_args


def draw_result(img_path, result_dict, data_names, save_result=True, save_path='./detect_results'):

    if save_result:
        import cv2, random
        from mindyolo.data import COCO80_TO_COCO91_CLASS
        os.makedirs(save_path, exist_ok=True)
        save_result_path = os.path.join(save_path, img_path.split('/')[-1])
        im = cv2.imread(img_path)
        category_id, bbox, score = result_dict['category_id'], result_dict['bbox'], result_dict['score']
        for i in range(len(bbox)):
            # draw box
            x_l, y_t, w, h = bbox[i][:]
            x_r, y_b = x_l + w, y_t + h
            x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
            _color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(_color), 2)

            # draw label
            class_name_index = COCO80_TO_COCO91_CLASS.index(category_id[i])
            class_name = data_names[class_name_index] # args.data.names[class_name_index]
            text = f'{class_name}: {score[i]}'
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(_color), -1)
            cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # save results
        cv2.imwrite(save_result_path, im)


if __name__ == '__main__':
    args = parse_args('infer')
    enginer = Enginer(args, 'infer')
    result_dict = enginer.detect(args.image_path)
    draw_result(args.image_path, result_dict, args.data.names, save_result=True, save_path='./detect_results')
