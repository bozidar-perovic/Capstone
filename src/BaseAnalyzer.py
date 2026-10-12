from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        pass
