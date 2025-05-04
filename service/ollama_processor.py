import base64
import io
import os
from typing import Tuple, Union

import ollama
from PIL.Image import Image

from processor import ImageProcessor

prompt = os.environ.get('OLLAMA_PROMPT', 'Describe this image in detail')

class OllamaImageProcessor(ImageProcessor):

    def __init__(self):
        self._models_cache = {model['model'] for model in ollama.list()['models']}

    def can_process(self, model_name: str) -> bool:
        return model_name in self._models_cache

    def generate_caption(self, model_name: str, image: Image) -> Tuple[str, str]:
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

            response = ollama.generate(model=model_name,
                                       # TODO externalize prompt to API - user defined/ENV - default/hardcoded default
                                       prompt=prompt,
                                       images=[base64_image])
            if response and response.response:
                return 'ok', response.response.strip()
            else:
                return 'error', 'No response from Ollama'

        except Exception as e:
            return 'error', f'Ollama processing error: {str(e)}'
