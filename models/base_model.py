from abc import ABC, abstractmethod
import torch

class BaseModel():
    def __init__(self, opt, manager):
        super(BaseModel, self).__init__()
        self.opt = opt
        # Can use manager here
        self.manager = manager
        self.logger = manager.get_logger()

        # Setup, only support cuda device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise EnvironmentError()