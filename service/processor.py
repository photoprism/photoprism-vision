from abc import ABC, abstractmethod

from PIL.Image import Image

from api import Labels, NSFW


class ImageProcessor(ABC):
    @abstractmethod
    def can_process(self, model_name: str, model_version: str) -> bool:
        pass

    @abstractmethod
    def generate_caption(self, model_name: str, model_version: str, image: Image) -> tuple[str, str]:
        """
        :param model_name: name of the requested model
        :param model_version: version of the requested model
        :param image: Image object
        """
        pass

    @abstractmethod
    def generate_labels(self, model_name: str, model_version: str, images: list[Image]) -> tuple[str, Labels | str]:
        """
        :param model_name: name of the requested model
        :param model_version: version of the requested model
        :param images: Image objects
        """
        pass

    @abstractmethod
    def detect_nsfw(self, model_name: str, model_version: str, image: Image) -> tuple[str, NSFW | str]:
        """
        :param model_name: name of the requested model
        :param model_version: version of the requested model
        :param image: Image object
        """
        pass
