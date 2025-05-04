import base64
import io

import requests
from PIL import Image
from PIL.Image import Image as ImageType


def load_image(url: str) -> ImageType:
    return Image.open(requests.get(url, stream=True).raw).convert('RGB')


def decode_image(base64image: str) -> ImageType:
    image_data = base64image.split(',')
    if len(image_data) == 2:
        image_type, image_data = image_data
        return Image.open(io.BytesIO(base64.b64decode(image_data)))
    else:
        raise ValueError('Invalid base64 image format')
