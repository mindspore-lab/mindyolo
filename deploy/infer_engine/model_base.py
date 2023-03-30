from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class ModelBase(metaclass=ABCMeta):
    """
    base class for model load and infer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def infer(self, input: np.numarray) -> List[np.numarray]:
        """
        model inference, just for single input
        Args:
            input: np img

        Returns:

        """
        pass

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model
