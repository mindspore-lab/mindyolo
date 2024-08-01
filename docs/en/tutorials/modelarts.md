

#MindYOLO ModelArts Training Quick Start

This article mainly introduces the training method of MindYOLO using the [ModelArts](https://www.huaweicloud.com/product/modelarts.html) platform.
ModelArts related tutorial reference [Help Center](https://docs.xckpjs.com/zh-cn/modelarts/index.html)

## Prepare data and code
Use the OBS service to upload data sets. For related operation tutorials, see [OBS User Guide](https://docs.xckpjs.com/zh-cn/obs/index.html) to obtain the [AK] of this account (https:// docs.xckpjs.com/zh-cn/browsertg/obs/obs_03_1007.html), please consult the corresponding platform administrator or account person in charge for the server address. If the AK is not in the location specified in the user guide, please also consult the platform administrator or account person in charge. . <br>
operate:


1. Log in to obs browser+
 ![obs](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/obs.jpg)
2. Create a bucket -> create a new folder (eg: coco)
 ![Bucket](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E5%88%9B%E5%BB%BA%E6%A1%B6.jpg)
3. To upload data files, please place the data files in a separate folder (that is, coco in the use case). The code will copy the data in the obs bucket, and the copied content will be all files in this folder (such as coco). document. Without creating a new folder, you cannot select the complete data set.

 ![Dataset](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)

## Prepare code

Also use the OBS service to upload the training code. <br>
Operation: Create a bucket -> Create a new folder (such as: mindyolo) -> Upload the code file, create an output folder at the same level of mindyolo to store training records, and create a log folder to store logs.
 ![Bucket directory](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E4%BB%A3%E7%A0%81%E6%A1%B6.jpg)
 ![Kit code](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E5%A5%97%E4%BB%B6%E4%BB%A3%E7%A0%81.jpg)


## Create new algorithm

1. Select Algorithm Management->Create in the tab.
 ![Create algorithm](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E6%96%B0%E5%BB%BA%E7%AE%97%E6%B3%95.jpg)
2. Customize the algorithm name, select Ascend-Powered-Engine for the prefabricated framework, select the MindSpore-2.0 version image for the master branch, and select the MindSpore-1.8.1 version image for the r0.1 branch. Set the code directory, startup file, input, and output. and superparameters.
 ![Algorithm configuration](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E7%AE%97%E6%B3%95%E9%85%8D%E7%BD%AE.jpg)


* If you need to load pre-trained weights, you can select the uploaded model file in the model selection and add the ckpt_dir parameter in the running parameters.
 ![ckpt](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/ckpt.jpg)
* The startup file is train.py
* To run super parameters, enable_modelarts needs to be added, and the value is True.
* The running super parameter config path refers to the directory of the running environment preview in the training job, such as /home/ma-user/modelarts/user-job-dir/mindyolo/configs/yolov5/yolov5n.yaml
* If distributed training scenarios are involved, the hyperparameter is_parallel needs to be added and set to True when running in distributed mode and False when running on a single card.

## Create new job
1. Select in the ModelArts service: Training Management -> Training Jobs -> Create a training job, set the job name, and choose not to include it in the experiment; Create Method -> My Algorithm, select the newly created algorithm;
 ![task](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E5%88%9B%E5%BB%BA%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A.jpg)
 ![task1](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A1.jpg)
2. Training input -> Data storage location, select the obs data bucket just created (coco in the example), select the output folder when preparing the code for training output, and set the config hyperparameter value according to the running environment preview;
 ![task2](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A2.jpg)
3. Select the resource pool, specifications, computing nodes, and select the log folder when creating the code for the job log path.
 ![task3](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E8%AE%AD%E7%BB%83%E4%BD%9C%E4%B8%9A3.jpg)
 ![Specifications](https://github.com/Ash-Lee233/image/raw/main/mindyolo/modelarts_tutorial/%E8%A7%84%E6%A0%BC.jpg)
4. Submit training and it will be running after queuing.

## Modify job
Select Rebuild on the training job page to modify the selected job configuration.



