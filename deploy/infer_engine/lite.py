import numpy as np
from .model_base import ModelBase

class LiteModel(ModelBase):
    def __init__(self, model_path, device_id = 0):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

    def _init_model(self):
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = ['ascend']
        context.ascend.device_id = self.device_id

        self.model = mslite.Model()
        self.model.build_from_from_file(self.model_path, mslite.ModelType.MINDIR, context)

    def infer(self, input):
        inputs = self.model.get_inputs()
        self.model.resize(inputs, [list(input.shape)])
        inputs[0].set_data_from_numpy(input)

        outputs = self.model.predict(inputs)
        outputs = [outputs.get_data_to_numpy() for output in outputs]
        return outputs
