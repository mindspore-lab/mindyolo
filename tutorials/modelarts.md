## MindYOLO Cloud Training Quick Start Guide
This article introduces MindYOLO training with the OPENI [Qizhi Platform](https://openi.pcl.ac.cn/). .

### External Project Migration

Click on the plus icon in the upper right corner of the main page and select "New Migration" to migrate MindYOLO from GitHub to the OPENI platform.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/newmigration.png"/>
</div>
Enter the URL of MindYOLO to initiate the migration.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img.png"/>
</div>


### Prepare the Dataset

You can upload your own dataset or linked existing datasets on the platform.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_1.png"/>
</div>
When uploading a personal dataset, select NPU as the cluster option.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_2.png"/>
</div>

### Prepare Pretrained Models (Optional)

If you want to load pretrained weights, you can add them in the "Models" tab.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_3.png"/>
</div>
When importing local models, specify the framework as MindSpore.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_4.png"/>
</div>

### Create a New Training Task

Select "Train Task", and then click "New Train Task."
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_5.png"/>
</div>
In the "Basic Info" section, select Ascend NPU as the computing resources.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_6.png"/>
</div>
Set the parameters and add the runtime parameters.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_7.png"/>
</div>

* If you want to load pretrained weights, select the uploaded model file in the "Model" section, and add the "ckpt_dir" parameter in the runtime parameters. The parameter value should be "/cache/*.ckpt" where * represents the actual filename.
* In the AI Engine section, select MindSpore-1.8.1-aarch64, and set the start file as train.py.
* Add the "enable_modelarts" parameter with a value of True in the runtime parameters.
* Specify the specific model in the "config" parameter in the runtime parameters. The parameter value prefix should be "/home/work/user-job-dir/" followed by the runtime version number. The runtime version number is typically V0001 for a new training task.

In a distributed training scenario, select the desired number of cards in the "Specifications" section and add the "is_parallel" parameter with a value of True in the runtime parameters.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_12.png"/>
</div>

### Modify an Existing Training Task

Click on the "Modify" button for an existing training task to make parameter modifications based on the existing task and run a new training task.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_8.png"/>
</div>

Note: The runtime version number should be the parents version number + 1.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_9.png"/>
</div>

### Check Status

Click on the corresponding task name to view configuration information, logs, resource usage, and download results.
<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_10.png"/>
</div>

<div align=center>
<img width='600' src="https://github.com/chenyang23333/images/raw/main/img_11.png"/>
</div>
