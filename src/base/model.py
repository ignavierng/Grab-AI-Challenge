from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all models
    """
    def __init__(self, params):
        self.params = params
        self.fitted = False    # Keep track if model is trained

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def predict(self, X_np, y_np):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def print_summary(self, print_func):
        pass

    @property
    def logger(self):
        """
        Each class that inherits Model needs to define _logger as class attribute
        """
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')
