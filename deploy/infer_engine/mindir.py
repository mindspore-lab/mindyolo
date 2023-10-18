"""MindSpore Graph Mode Inference"""

from .model_base import ModelBase
import numpy as np


class MindIRModel(ModelBase):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        global ms, nn, Tensor
        from mindspore import Tensor
        import mindspore.nn as nn
        import mindspore as ms
        ms.set_context(mode=ms.GRAPH_MODE)
        self.model = nn.GraphCell(ms.load(self.model_path))
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

    def infer(self, input):
        inputs = Tensor(input)
        outputs = self.model(inputs)
        # extract result from func return into Tensor lists
        output_list = []
        for output in outputs:
            if isinstance(output, tuple):
                for out in output:
                    assert not isinstance(out, tuple), 'only support level one tuple'
                    output_list.append(out.asnumpy())
            else:
                output_list.append(output.asnumpy())
        return output_list
