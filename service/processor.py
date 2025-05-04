from abc import ABC, abstractmethod
from typing import Tuple, Union

from PIL.Image import Image


class ImageProcessor(ABC):
    @abstractmethod
    def can_process(self, model_name: str) -> bool:
        pass

    @abstractmethod
    def generate_caption(self, model_name: str, image: Image) -> Tuple[str, str]:
        """
        :param model_name: name of requested model
        :param image: Image object
        """
        pass

    @abstractmethod
    def generate_labels(self, model_name: str, image: Image) -> Tuple[str, str]:
        """
        :param model_name: name of requested model
        :param image: Image object
        """
        pass
