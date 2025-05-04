import base64
import io
import logging
import os
from typing import Any, Tuple, override

import ollama
from PIL.Image import Image

from processor import ImageProcessor
from api import Labels

logger = logging.getLogger(__name__)

caption_prompt = os.environ.get('OLLAMA_CAPTION_PROMPT', 'Describe this image in detail')
labels_prompt = os.environ.get('OLLAMA_LABELS_PROMPT', 'Generate from 1 to 5 worded labels for this image')


class OllamaImageProcessor(ImageProcessor):
    def __init__(self):
        self._models_cache = {model['model'] for model in ollama.list()['models']}

    def can_process(self, model_name: str) -> bool:
        return model_name in self._models_cache

    @override
    def generate_caption(self, model_name: str, image: Image) -> Tuple[str, str]:
        return self._generate_with_prompt(model_name, image, caption_prompt)

    @override
    def generate_labels(self, model_name: str, image: Image) -> Tuple[str, Labels | str]:
        schema = Labels.model_json_schema()
        status, result = self._generate_with_prompt(model_name, image, labels_prompt, schema=schema)
        if status == 'ok':
            try:
                labels = Labels.model_validate_json(result)
                return status, labels
            except Exception as e:
                return 'error', f'Failed to parse labels JSON: {str(e)}'
        return status, result

    def _generate_with_prompt(self, model_name: str, image: Image, prompt: str, schema=None) -> Tuple[str, Any]:
        try:
            base64_image = self._convert_image_to_base64(image)
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                images=[base64_image],
                format=schema
            )
            return self._process_ollama_response(response)
        except Exception as e:
            return 'error', f'Ollama processing error: {str(e)}'

    @staticmethod
    def _convert_image_to_base64(image: Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        return base64.b64encode(img_byte_arr).decode('utf-8')

    @staticmethod
    def _process_ollama_response(response) -> Tuple[str, str]:
        logger.debug(response)

        if response and response.response:
            return 'ok', response.response.strip()
        return 'error', 'No response from Ollama'
