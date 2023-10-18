"""ONNX Runtime Inference"""

from .model_base import ModelBase
import numpy as np


class ONNXRuntimeModel(ModelBase):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        global ort
        import onnxruntime as ort

        self.model = ort.InferenceSession(self.model_path, providers=ort.get_available_providers())
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")

    def infer(self, input):
        assert len(self.model.get_inputs()) == 1, \
            "Input shape should be 1 but got {}".format(len(self.model.get_inputs()))
        input_name = self.model.get_inputs()[0].name
        # extract result from func return into Tensor lists
        output_list = self.model.run(None, {input_name: input})
        return output_list
