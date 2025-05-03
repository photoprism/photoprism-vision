import base64
import io
from typing import Tuple

import ollama
from PIL.Image import Image


def ollama_caption(image: Image) -> Tuple[str, str]:
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        response = ollama.generate(model='llava',
                                   prompt='Describe this image in detail',
                                   images=[base64_image])

        if response and response.response:
            return 'ok', response.response.strip()
        else:
            return 'error', 'No response from Ollama'

    except Exception as e:
        return 'error', f'Ollama processing error: {str(e)}'
