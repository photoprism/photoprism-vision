import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, List, Tuple

from flask import Flask, Response, jsonify, request

from local_processor import MODEL_CONFIG
from local_processor import LocalImageProcessor
from ollama_processor import OllamaImageProcessor
from api import ApiResponse, Caption, Model, Text
from processor import ImageProcessor
from utils import decode_image, load_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
image_processors: List[ImageProcessor] = [LocalImageProcessor()]

if os.getenv('OLLAMA_ENABLED', 'false').lower() == 'true':
    image_processors.append(OllamaImageProcessor())


def create_response(data: Any, status_code: int = HTTPStatus.OK) -> Tuple[Response | str, int]:
    if isinstance(data, ApiResponse):
        return data.model_dump_json(), status_code
    return jsonify(data), status_code


def parse_image_from_request():
    data = request.get_json() if request.is_json else request.args
    image = None
    if data.get('url'):
        image = load_image(data['url'])
    elif data.get('images'):
        image = decode_image(data['images'][0])
    return data, image


@app.route('/api/v1/vision/caption', methods=['POST', 'GET'])
def default_process_image_caption() -> Tuple[Response, int]:
    return process_image_caption("kosmos-2")


@app.route('/api/v1/vision/caption/<model_name>', methods=['POST', 'GET'])
def process_image_caption(model_name: str) -> Tuple[Response, int]:
    try:
        data, image = parse_image_from_request()
        if not image:
            return create_response({'error': "image or url missing"}, HTTPStatus.BAD_REQUEST)

        for processor in image_processors:
            if processor.can_process(model_name):
                status, result = processor.generate_caption(model_name, image)
                if status == 'ok':
                    response_data = ApiResponse(
                        id=data.get('id', str(uuid.uuid4())),
                        result=Caption(caption=Text(text=result)),
                        model=Model(
                            name=model_name,
                            version=MODEL_CONFIG['MODELS'].get(model_name, {}).get('version', 'latest')
                        ),
                    )
                    return create_response(response_data, HTTPStatus.OK)
                return create_response({'error': result}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return create_response({'error': f"There is no image processor that has {model_name} available."},
                               HTTPStatus.BAD_REQUEST)
    except Exception as e:
        return create_response({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


@app.route('/api/v1/vision/labels/<model_name>', methods=['POST', 'GET'])
def process_image_labels(model_name: str) -> Tuple[Response, int]:
    try:
        data = request.get_json() if request.is_json else request.args
        images = []
        if data.get('url'):
            images = [load_image(data['url'])]
        elif data.get('images'):
            images = decode_image(data['images'])

        if not images:
            return create_response({'error': "images or url missing"}, HTTPStatus.BAD_REQUEST)

        for processor in image_processors:
            if processor.can_process(model_name):
                status, result = processor.generate_labels(model_name, images)
                if status == 'ok':
                    response_data = ApiResponse(
                        id=data.get('id', str(uuid.uuid4())),
                        result=result,
                        model=Model(
                            name=model_name,
                            version=MODEL_CONFIG['MODELS'].get(model_name, {}).get('version', 'latest')
                        ),
                    )
                    return create_response(response_data, HTTPStatus.OK)
                return create_response({'error': result}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return create_response({'error': f"There is no images processor that has {model_name} available."},
                               HTTPStatus.BAD_REQUEST)
    except Exception as e:
        return create_response({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


@app.route('/api/v1/vision/nsfw/<model_name>', methods=['POST', 'GET'])
def detect_nsfw(model_name: str) -> Tuple[Response, int]:
    try:
        data, image = parse_image_from_request()
        if not image:
            return create_response({'error': "image or url missing"}, HTTPStatus.BAD_REQUEST)

        for processor in image_processors:
            if processor.can_process(model_name):
                status, result = processor.detect_nsfw(model_name, image)
                if status == 'ok':
                    response_data = ApiResponse(
                        id=data.get('id', str(uuid.uuid4())),
                        result=result,
                        model=Model(
                            name=model_name,
                            version=MODEL_CONFIG['MODELS'].get(model_name, {}).get('version', 'latest')
                        ),
                    )
                    return create_response(response_data, HTTPStatus.OK)
                return create_response({'error': result}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return create_response({'error': f"There is no image processor that has {model_name} available."},
                               HTTPStatus.BAD_REQUEST)
    except Exception as e:
        return create_response({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
