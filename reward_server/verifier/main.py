from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

# Type definitions
T = TypeVar('T')
VerifyResult = Dict[str, Any]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Verifier:
    """Registry for all verifier classes."""
    _verifiers: Dict[str, Type['BaseVerifier']] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a verifier class."""

        def decorator(verifier_cls: Type['BaseVerifier']) -> Type['BaseVerifier']:
            verifier_name = name or verifier_cls.__name__
            cls._verifiers[verifier_name] = verifier_cls
            return verifier_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type['BaseVerifier']]:
        """Get a verifier class by name."""
        return cls._verifiers.get(name)

    @classmethod
    def list_verifiers(cls) -> List[str]:
        """List all registered verifiers."""
        return list(cls._verifiers.keys())


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    def __init__(
        self,
        is_training: bool,
        step: int,
        total_steps: int,
        query: Optional[List] = None,
        image_path: Optional[List[str]] = None,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        verifier_style: str = 'rule',
        det_verifier_normalized: bool = False,
        det_reward_ratio: Dict[str, float] = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        is_training : bool
            true: training mode, false: evaluation mode
        step : int
            current step
        total_steps : int
            total steps
        query : Optional[List], optional
            query of the current data, without chat format
        image_path : Optional[List[str]], optional
            image paths of the current data, it should be a global path, it should be a list
        image_grid_thw : Optional[List[Tuple[int, int, int]]], optional
            image grid of the current data, it should be a list of tuple, each tuple contains (times - t, height - h, width - w)
        verifier_style : str, optional
            either 'rule' or 'model', by default 'rule'
        det_verifier_normalized : bool, optional
            whether the detection verifier is normalized
        det_reward_ratio : Dict[str, float], optional
            reward ratio of the detection verifier
        """

        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps
        self.step_ratio = float(step) / float(total_steps)
        self.image_path = image_path
        self.image_grid_thw = image_grid_thw
        self.query = query
        self.verifier_style = verifier_style
        self.det_verifier_normalized = det_verifier_normalized
        self.det_reward_ratio = det_reward_ratio

    @abstractmethod
    def verify_format(self, predict_str: Any) -> VerifyResult:
        """
        Verify the format of the input.
        
        Args:
            predict_str: The input to verify

        Returns:
            Dict containing verification results
        """
        pass

    @abstractmethod
    def verify_accuracy(self, predict_str: Any, solution: Any) -> VerifyResult:
        """
        Verify the accuracy of the input against the solution.
        
        Args:
            predict_str: The input to verify
            solution: The solution to verify against
            
        Returns:
            Dict containing verification results
        """
        pass

    def verify_length(self, predict_str: Any, answer: Any) -> float:
        """
        Verify the length of the input.
        """

        logger.warning("length verification is not implemented! Return 0.0")
        del predict_str, answer
        return 0.0
