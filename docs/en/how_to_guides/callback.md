# Usage of MindYOLO callback function

**Callback function**: When the program runs to a certain mount point, all methods registered to the mount point at runtime will be automatically called.
The flexibility and extensibility of the program can be increased by using the callback function, because users can register custom methods to the mount point to be called without modifying the code in the program.

In MindYOLO, the callback function is specifically implemented in the [mindyolo/utils/callback.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/callback.py) file.
```python
#mindyolo/utils/callback.py
@CALLBACK_REGISTRY.registry_module()
class callback_class_name(BaseCallback):
    def __init__(self, **kwargs):
        super().__init__()
...
def callback_fn_name(self, run_context: RunContext):
    pass
```

Add a dictionary list under the callback field of the model's yaml file to implement the call
```yaml
#Callback function configuration dictionary:
callback:
- { name: callback_class_name, args: xx }
- { name: callback_class_name2, args: xx }
```
For example, take YOLOX as an example:

Add logic to the on_train_step_begin method in the YoloxSwitchTrain class in the mindyolo/utils/callback.py file to print "train step begin” log
```python
@CALLBACK_REGISTRY.registry_module()
class YoloxSwitchTrain(BaseCallback):

    def on_train_step_begin(self, run_context: RunContext):
        # Custom logic
        logger.info("train step begin")
        pass

```
Add the callback function under the callback field of the YOLOX corresponding yaml file configs/yolox/hyp.scratch.yaml
```yaml
callback:
  - { name: YoloxSwitchTrain, switch_epoch_num: 285 }
```
Then the logger.info("train step begin") statement will be executed before each training step is executed.

With the help of the callback function, users can customize the logic that needs to be executed at a certain mount point without having to understand the code of the complete training process.