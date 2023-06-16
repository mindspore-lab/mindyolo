# MindYOLO ModelArts训练快速入门

本文主要介绍MindYOLO借助[ModelArts](https://www.huaweicloud.com/product/modelarts.html)平台的训练方法。
ModelArts相关教程参考[帮助中心](https://docs.xckpjs.com/zh-cn/modelarts/index.html)

## 准备数据及代码
使用OBS服务上传数据集，相关操作教程见[OBS用户指南](https://docs.xckpjs.com/zh-cn/obs/index.html)，获取本账户的[AK](https://docs.xckpjs.com/zh-cn/browsertg/obs/obs_03_1007.html)，服务器地址请咨询对应平台管理员或账号负责人，如AK不在用户指南指定位置，也请咨询平台管理员或账号负责人。<br>
操作：
1. 登录obs browser+
![obs](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/obs.jpg)
2. 创建桶 -> 新建文件夹（如：coco）
![桶](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E5%88%9B%E5%BB%BA%E6%A1%B6.jpg)
3. 上传数据文件，请将数据文件统一单独放置在一个文件夹内（即用例中的coco），代码中会对obs桶内数据做拷贝，拷贝内容为此文件夹（如：coco）下所有的文件。如未新建文件夹，就无法选择完整数据集。
![数据集](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)


## 准备代码

同样使用OBS服务上传训练代码。<br>
操作：创建桶 -> 新建文件夹（如：mindyolo）-> 上传代码文件，在mindyolo同层级下创建output文件夹用于存放训练记录，创建log文件夹用于存放日志。
![桶目录](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E4%BB%A3%E7%A0%81%E6%A1%B6.jpg)
![套件代码](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E5%A5%97%E4%BB%B6%E4%BB%A3%E7%A0%81.jpg)


## 新建算法

1. 在选项卡中选择算法管理->创建。
![创建算法](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E6%96%B0%E5%BB%BA%E7%AE%97%E6%B3%95.jpg)
2. 自定义算法名称，预制框架选择Ascend-Powered-Engine，master分支请选择MindSpore-2.0版本镜像，r0.1分支请选择MindSpore-1.8.1版本镜像，设置代码目录、启动文件、输入、输出以及超参。
![算法配置](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E7%AE%97%E6%B3%95%E9%85%8D%E7%BD%AE.jpg)


* 如需加载预训练权重，可在选择模型中选择已上传的模型文件，并在运行参数中增加ckpt_dir参数
![ckpt](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/ckpt.jpg)
* 启动文件为train.py
* 运行超参需添加enable_modelarts，值为True
* 运行超参config路径参考训练作业中运行环境预览的目录，如/home/ma-user/modelarts/user-job-dir/mindyolo/configs/yolov5/yolov5n.yaml
* 如涉及分布式训练场景，需增加超参is_parallel，并在分布式运行时设置为True，单卡时为False

## 新建作业
1. 在ModelArts服务中选择：训练管理 -> 训练作业 -> 创建训练作业，设置作业名称，选择不纳入实验；创建方式->我的算法选择刚才新建的算法；
![task](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E5%88%9B%E5%BB%BA%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A.jpg)
![task1](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A1.jpg)
2. 训练输入->数据存储位置，选择刚才创建的obs数据桶（示例中为coco），训练输出选择准备代码时的output文件夹，并根据运行环境预览设置好config超参值;
![task2](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A2.jpg)
3. 选择资源池、规格、计算节点，作业日志路径选择创建代码时的log文件夹
![task3](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A3.jpg)
![规格](https://github.com/Ash-Lee233/image/blob/main/mindyolo/modelarts_tutorial/%E8%A7%84%E6%A0%BC.jpg)
4. 提交训练，排队后会进入运行中

## 修改作业
在训练作业页面选择重建，可修改选择的作业配置
