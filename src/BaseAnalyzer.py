"""Abstract base class interface for test analyzers.

Defines the contract that all static flaky test analyzers (rule-based,
machine learning, and adaptive hybrid) must implement to ensure interchangeable
usage across evaluation pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):
    """Abstract base analyzer class.

    All analyzers in this project must inherit from BaseAnalyzer and provide
    an implementation of the analyze method.
    """

    requires_training: bool = False

    def train(self, train_data: Any) -> "BaseAnalyzer":
        """Prepare the analyzer; stateless analyzers need no training.

        Trainable subclasses must override this hook and set
        ``requires_training = True``.
        """
        if self.requires_training:
            raise NotImplementedError(
                "Trainable analyzers must implement train."
            )
        return self

    def get_routing_summary(self) -> dict:
        """Return optional routing diagnostics after analysis."""
        return {}

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """Analyze test cases and produce continuous flakiness risk scores.

        Args:
            data: Input test data, typically a pandas DataFrame containing
                at least a 'full_code' column with Java method source strings,
                or a file path to a CSV dataset.

        Returns:
            A pandas DataFrame with input test data and an appended
            'flaky_score' column representing continuous predicted risk
            between 0.0 and 1.0.

        Raises:
            ValueError: If the analyzer requires pre-training and has not
                yet been fitted on training data.
        """
        pass
