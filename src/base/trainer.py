from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    Abstract base class for all trainers
    """

    def __init__(self, model, dataset, params, output_dir):
        self.model = model
        self.dataset = dataset
        self.params = params
        self.output_dir = output_dir

    @abstractmethod
    def train(self):
        pass
