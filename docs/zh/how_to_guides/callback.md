# MindYOLO回调函数用法

**回调函数**：当程序运行到某个挂载点时，会自动调用在运行时注册到该挂载点的所有方法。
通过回调函数的形式可以增加程序的灵活性和扩展性，因为用户可以将自定义方法注册到要调用的挂载点，而无需修改程序中的代码。

在MindYOLO中，回调函数具体实现在[mindyolo/utils/callback.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/callback.py)文件中。
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

通过模型的yaml文件callback字段下添加一个字典列表来实现调用
```yaml
#回调函数配置字典：
callback:
  - { name: callback_class_name, args: xx }
  - { name: callback_class_name2, args: xx }
```
例如以YOLOX为示例：

在mindyolo/utils/callback.py文件YoloxSwitchTrain类中on_train_step_begin方法里面添加逻辑，打印“train step begin”的日志
```python
@CALLBACK_REGISTRY.registry_module()
class YoloxSwitchTrain(BaseCallback):

    def on_train_step_begin(self, run_context: RunContext):
         # 自定义逻辑
        logger.info("train step begin")
        pass

```
YOLOX对应的yaml文件configs/yolox/hyp.scratch.yaml的callback字段下添加该回调函数
```yaml
callback:
  - { name: YoloxSwitchTrain, switch_epoch_num: 285 }
```
则每个训练step执行前都会执行logger.info("train step begin")语句。

借助回调函数，用户可以自定义某个挂载点需要执行的逻辑，而无需理解完整的训练流程的代码。