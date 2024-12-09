# MindYOLO Benchmark and Baselines

## Detection
<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |    37.2     | 3.2M   | [yaml](./configs/yolov8/yolov8n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |    44.6     | 11.2M  | [yaml](./configs/yolov8/yolov8s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt)                 |
| YOLOv8 | M                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 25.9M  | [yaml](./configs/yolov8/yolov8m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt)                 |
| YOLOv8 | L                  |  16 * 8   |    640    | MS COCO 2017 |    52.8     | 43.7M  | [yaml](./configs/yolov8/yolov8l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt)                 |
| YOLOv8 | X                  |  16 * 8   |    640    | MS COCO 2017 |    53.7     | 68.2M  | [yaml](./configs/yolov8/yolov8x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt)                 |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |    37.5     | 6.2M   | [yaml](./configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt)              |
| YOLOv7 | L                  |  16 * 8   |    640    | MS COCO 2017 |    50.8     | 36.9M  | [yaml](./configs/yolov7/yolov7.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt)                   |
| YOLOv7 | X                  |  12 * 8   |    640    | MS COCO 2017 |    52.4     | 71.3M  | [yaml](./configs/yolov7/yolov7-x.yaml)       | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt)                 |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |    27.3     | 1.9M   | [yaml](./configs/yolov5/yolov5n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |    37.6     | 7.2M   | [yaml](./configs/yolov5/yolov5s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)                  |
| YOLOv5 | M                  |  32 * 8   |    640    | MS COCO 2017 |    44.9     | 21.2M  | [yaml](./configs/yolov5/yolov5m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)                  |
| YOLOv5 | L                  |  32 * 8   |    640    | MS COCO 2017 |    48.5     | 46.5M  | [yaml](./configs/yolov5/yolov5l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)                  |
| YOLOv5 | X                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 86.7M  | [yaml](./configs/yolov5/yolov5x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt)                  |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |    45.4     | 27.6M  | [yaml](./configs/yolov4/yolov4.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93.ckpt)      |
| YOLOv4 | CSPDarknet53(silu) |  16 * 8   |    608    | MS COCO 2017 |    45.8     | 27.6M  | [yaml](./configs/yolov4/yolov4-silu.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |    45.5     | 61.9M  | [yaml](./configs/yolov3/yolov3.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)         |
| YOLOX  | N                  |   8 * 8   |    416    | MS COCO 2017 |    24.1     | 0.9M   | [yaml](./configs/yolox/yolox-nano.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3.ckpt)                  |
| YOLOX  | Tiny               |   8 * 8   |    416    | MS COCO 2017 |    33.3     | 5.1M   | [yaml](./configs/yolox/yolox-tiny.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e.ckpt)               |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |    40.7     | 9.0M   | [yaml](./configs/yolox/yolox-s.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f.ckpt)                  |
| YOLOX  | M                  |   8 * 8   |    640    | MS COCO 2017 |    46.7     | 25.3M  | [yaml](./configs/yolox/yolox-m.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee.ckpt)                  |
| YOLOX  | L                  |   8 * 8   |    640    | MS COCO 2017 |    49.2     | 54.2M  | [yaml](./configs/yolox/yolox-l.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80.ckpt)                  |
| YOLOX  | X                  |   8 * 8   |    640    | MS COCO 2017 |    51.6     | 99.1M  | [yaml](./configs/yolox/yolox-x.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90.ckpt)                  |
| YOLOX  | Darknet53          |   8 * 8   |    640    | MS COCO 2017 |    47.7     | 63.7M  | [yaml](./configs/yolox/yolox-darknet53.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9.ckpt)          |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv10 | N                 |  32 * 8   |    640    | MS COCO 2017 |     38.3    | 513.63  | 2.8M   | [yaml](./configs/yolov10/yolov10n.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10n_500e_mAP383-c973023d.ckpt)                 |
| YOLOv10 | S                 |  32 * 8   |    640    | MS COCO 2017 |     45.7    | 503.38  | 8.2M   | [yaml](./configs/yolov10/yolov10s.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10s_500e_mAP457-8660fa84.ckpt)                 |
| YOLOv10 | M                 |  32 * 8   |    640    | MS COCO 2017 |     50.7    | 560.81  | 16.6M  | [yaml](./configs/yolov10/yolov10m.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10m_500e_mAP507-1cc8c5fb.ckpt)                 |
| YOLOv10 | B                 |  32 * 8   |    640    | MS COCO 2017 |     52.0    | 695.69  | 20.6M  | [yaml](./configs/yolov10/yolov10b.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10b_500e_mAP520-0b560f87.ckpt)                 |
| YOLOv10 | L                 |  32 * 8   |    640    | MS COCO 2017 |     52.6    | 782.61  | 25.9M  | [yaml](./configs/yolov10/yolov10l.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10l_500e_mAP526-226baf5f.ckpt)                 |
| YOLOv10 | X                 |  20 * 8   |    640    | MS COCO 2017 |     53.7    | 650.63  | 31.8M  | [yaml](./configs/yolov10/yolov10l.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov10/yolov10x_500e_mAP537-aaaa57bb.ckpt)                 |
| YOLOv9 | T                  |  16 * 8   |    640    | MS COCO 2017 |     37.3    | 350  | 2.0M   | [yaml](./configs/yolov9/yolov9-t.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9t_500e_MAP373-c0ee5cbc.ckpt)                 |
| YOLOv9 | S                  |  16 * 8   |    640    | MS COCO 2017 |     46.3    | 377  | 7.1M   | [yaml](./configs/yolov9/yolov9-s.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9s_500e_MAP463-b3cb691d.ckpt)                 |
| YOLOv9 | M                  |  16 * 8   |    640    | MS COCO 2017 |     51.4    | 499  | 20.0M   | [yaml](./configs/yolov9/yolov9-m.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9m_500e_MAP514-86aa8761.ckpt)                 |
| YOLOv9 | C                  |  16 * 8   |    640    | MS COCO 2017 |     52.6    | 627  | 25.3M   | [yaml](./configs/yolov9/yolov9-c.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9c_500e_MAP526-ff7bdf68.ckpt)                 |
| YOLOv9 | E                  |  16 * 8   |    640    | MS COCO 2017 |     55.1    | 826  | 57.3M   | [yaml](./configs/yolov9/yolov9-e.yaml)        | [ [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov9/yolov9e_500e_MAP551-6b55c121.ckpt)                 |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |     37.3    | 373.55  | 3.2M   | [yaml](./configs/yolov8/yolov8n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-0e737186-910v2.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |     44.7    | 365.53  | 11.2M  | [yaml](./configs/yolov8/yolov8s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-fae4983f-910v2.ckpt)  |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |     37.5    | 496.21  | 6.2M   | [yaml](./configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-1d2ddf4b-910v2.ckpt)              |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |     27.4    | 736.08  | 1.9M   | [yaml](./configs/yolov5/yolov5n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-bedf9a93-910v2.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |     37.6    | 787.34  | 7.2M   | [yaml](./configs/yolov5/yolov5s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-df4a45b6-910v2.ckpt)                  |
| YOLOv5 | N6                 |  32 * 8   |    1280   | MS COCO 2017 |     35.7    | 1543.35 | 3.5M   | [yaml](./configs/yolov5/yolov5n6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n6_300e_mAP357-49d91077.ckpt)                  |
| YOLOv5 | S6                 |  32 * 8   |    1280   | MS COCO 2017 |     44.4    | 1514.98 | 13.6M  | [yaml](./configs/yolov5/yolov5s6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s6_300e_mAP444-aeaffe77.ckpt)                  |
| YOLOv5 | M6                 |  32 * 8   |    1280   | MS COCO 2017 |     51.1    | 1769.17 | 38.5M  | [yaml](./configs/yolov5/yolov5m6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5m6_300e_mAP511-025d9536.ckpt)                  |
| YOLOv5 | L6                 |  16 * 8   |    1280   | MS COCO 2017 |     53.6    | 894.65  | 82.9M  | [yaml](./configs/yolov5/yolov5l6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5l6_300e_mAP536-617a1cc1.ckpt)                  |
| YOLOv5 | X6                 |   8 * 8   |    1280   | MS COCO 2017 |     54.4    | 864.43  | 140.9M | [yaml](./configs/yolov5/yolov5x6.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5x6_300e_mAP545-81ebdca9.ckpt)                  |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |     46.1    | 337.25  | 27.6M  | [yaml](./configs/yolov4/yolov4.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-64b8506f-910v2.ckpt)      |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |     46.6    | 396.60  | 61.9M  | [yaml](./configs/yolov3/yolov3.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-81895f09-910v2.ckpt)         |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |     41.0    | 242.15  | 9.0M   | [yaml](./configs/yolox/yolox-s.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-cebd0183-910v2.ckpt)                   |
</details>

## Segmentation
<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name       | Scale | BatchSize | ImageSize | Dataset      | Box mAP (%) | Mask mAP (%) | Params |                Recipe                        | Download                                                                                                       |
|------------| :---: |   :---:   |   :---:   |--------------|    :---:    |     :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8-seg |   X   |  16 * 8   |    640    | MS COCO 2017 |     52.5    |     42.9     |  71.8M | [yaml](./configs/yolov8/seg/yolov8x-seg.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x-seg_300e_mAP_mask_429-b4920557.ckpt) |
</details>

## Depoly inference

- See [support list](./deploy/README.md)

## Notes
- Box mAP: Accuracy reported on the validation set.
