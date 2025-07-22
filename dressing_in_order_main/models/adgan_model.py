from .base_model import BaseModel

class AdganModel(BaseModel):
    def __init__(self, opt):
        super(AdganModel, self).__init__(opt)
        # initialize networks, load weights etc.
        self.opt = opt
        self.model_names = []
        self.visual_names = []
        self.loss_names = []

    def set_input(self, input):
        # save the input data for later processing
        self.input = input

    def forward(self):
        # run inference logic
        pass

    def optimize_parameters(self):
        # leave empty since you're probably testing/inference only
        pass