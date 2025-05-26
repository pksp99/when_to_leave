from abc import ABC, abstractmethod

class BaseModelApproach(ABC):
    @abstractmethod
    def predict(self, row:dict, override=False):
        pass

    @abstractmethod
    def prediction_keys(self):
        pass

    def _check_keys(self, row: dict) -> bool:
        return all(k in row and row[k] is not None for k in self.prediction_keys())