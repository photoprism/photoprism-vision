import logging
import os
from typing import override

from PIL.Image import Image
import numpy as np
from ultralytics import YOLO

from configuration import MODELS_DIRECTORY
from processor import ImageProcessor
from api import Label, Labels, NSFW

logger = logging.getLogger(__name__)

# Specified models will be loaded at startup
# see https://docs.ultralytics.com/models/ for available models and their configurations.
DETECTION_MODELS = os.environ.get('ULTRALYTICS_MODELS', 'yolo11n.pt').split(',')

CONFIDENCE_THRESHOLD = float(os.environ.get('ULTRALYTICS_CONFIDENCE_THRESHOLD', '0.25'))


class UltralyticsImageProcessor(ImageProcessor):
    def __init__(self):
        self._models = {}
        self._load_models()

    def _load_models(self):
        """Load default models if they exist"""
        for model in DETECTION_MODELS:
            try:
                self._models[model] = YOLO(f'{MODELS_DIRECTORY}/{model}')
                logger.info(f"Loaded detection model: {model}")
            except Exception as e:
                logger.warning(f"Failed to load detection model: {str(e)}")

    def can_process(self, model_name: str) -> bool:
        """Check if a model is available or can be loaded"""
        return model_name in self._models

    @override
    def generate_caption(self, model_name: str, image: Image) -> tuple[str, str]:
        """Generate a caption based on detected objects"""
        try:
            model = self._get_model(model_name)
            if model is None:
                return 'error', f'Model {model_name} not available'

            img_array = np.array(image)
            results = model(img_array, verbose=False)
            caption = self._results_to_caption(results)

            return 'ok', caption
        except Exception as e:
            return 'error', f'Ultralytics processing error: {str(e)}'

    @override
    def generate_labels(self, model_name: str, images: list[Image]) -> tuple[str, Labels | str]:
        """Generate labels for a list of images"""
        try:
            model = self._get_model(model_name)
            if model is None:
                return 'error', f'Model {model_name} not available'

            all_labels = []
            for image in images:
                img_array = np.array(image)
                results = model(img_array, verbose=False)

                image_labels = self._results_to_labels(results)
                all_labels.extend(image_labels)

            labels_obj = Labels(labels=all_labels)
            return 'ok', labels_obj
        except Exception as e:
            return 'error', f'Ultralytics processing error: {str(e)}'

    @override
    def detect_nsfw(self, model_name: str, image: Image) -> tuple[str, NSFW | str]:
        return 'error', f'UltralyticsImageProcessor does not support nsfw detection'

    def _get_model(self, model_name: str):
        """Get a model by name or load it if needed"""
        if model_name in self._models:
            return self._models[model_name]

        raise ValueError(f"Model {model_name} not found")

    @staticmethod
    def _results_to_caption(results) -> str:
        """Convert detection results to a descriptive caption"""
        if not results or len(results) == 0:
            return "No objects detected in the image."

        caption_parts = UltralyticsImageProcessor._results_to_labels(results)
        if len(caption_parts) == 1:
            caption = f"This image contains {caption_parts[0].name}."
        elif len(caption_parts) == 2:
            caption = f"This image contains {caption_parts[0].name} and {caption_parts[1].name}."
        else:
            caption = f"This image contains {', '.join([label.name for label in caption_parts[:-1]])}, and {caption_parts[-1].name}."

        return caption

    @staticmethod
    def _results_to_labels(results) -> list[Label]:
        """Convert detection results to a list of labels"""
        if not results or len(results) == 0:
            return []

        labels = []
        for result in results:
            # object detection model
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]
                    confidence = box.conf.item()

                    if confidence >= CONFIDENCE_THRESHOLD:
                        labels.append(
                            Label(name=class_name, priority=100 - int(confidence * 100.0), confidence=confidence))
            # classification model
            elif result.probs:
                for idx in result.probs.top5:
                    class_name = result.names[idx]
                    confidence = result.probs[idx].data.item()

                    if confidence >= CONFIDENCE_THRESHOLD:
                        labels.append(
                            Label(name=class_name, priority=100 - int(confidence * 100.0), confidence=confidence))

        return labels
