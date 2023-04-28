"""MindX SDK Inference"""

import numpy as np

from .model_base import ModelBase


class MindXModel(ModelBase):
    def __init__(self, model_path, device_id=0):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

        self._init_model()

    def _init_model(self):
        global base, Tensor
        from mindx.sdk import base, Tensor, visionDataFormat

        base.mx_init()
        self.model = base.model(self.model_path, self.device_id)
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

    def infer(self, input):
        inputs = Tensor(input)
        outputs = self.model.infer(inputs)
        list([output.to_host() for output in outputs])
        outputs = [np.array(output) for output in outputs]
        return outputs
