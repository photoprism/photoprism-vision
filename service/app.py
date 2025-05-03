import os
import uuid
from http import HTTPStatus
from typing import Tuple, Any

import requests
import torch
from PIL import Image
from flask import Flask, jsonify, request, Response
from transformers import (
    AutoProcessor, AutoModelForVision2Seq, VisionEncoderDecoderModel,
    ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
)

# Configuration Constants
MODEL_CONFIG = {
    'BASE_DIR': 'models',
    'MODELS': {
        'kosmos-2': {
            'path': 'models/kosmos-2-patch14-224',
            'source': 'microsoft/kosmos-2-patch14-224',
            'version': 'patch14-224'
        },
        'vit-gpt2': {
            'path': 'models/vit-gpt2-image-captioning',
            'source': 'nlpconnect/vit-gpt2-image-captioning',
            'version': 'latest'
        },
        'blip': {
            'path': 'models/blip-image-captioning-large',
            'source': 'Salesforce/blip-image-captioning-large',
            'version': 'latest'
        }
    }
}


class ModelManager:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self._initialize_models()

    def _initialize_models(self):
        os.makedirs(MODEL_CONFIG['BASE_DIR'], exist_ok=True)
        self._download_and_load_models()

    def _download_and_load_models(self):
        for model_name, config in MODEL_CONFIG['MODELS'].items():
            if not os.path.exists(config['path']):
                self._download_model(config['source'], config['path'])
            self._load_model(model_name, config['path'])

    @staticmethod
    def _download_model(source: str, path: str):
        print(f"Downloading {source}...")
        if 'kosmos' in source:
            AutoModelForVision2Seq.from_pretrained(source).save_pretrained(path)
            AutoProcessor.from_pretrained(source).save_pretrained(path)
        elif 'vit-gpt2' in source:
            VisionEncoderDecoderModel.from_pretrained(source).save_pretrained(path)
            ViTImageProcessor.from_pretrained(source).save_pretrained(path)
            AutoTokenizer.from_pretrained(source).save_pretrained(path)
        elif 'blip' in source:
            BlipForConditionalGeneration.from_pretrained(source).save_pretrained(path)
            BlipProcessor.from_pretrained(source).save_pretrained(path)

    def _load_model(self, model_name: str, path: str):
        if model_name == 'kosmos-2':
            self.models[model_name] = AutoModelForVision2Seq.from_pretrained(path)
            self.processors[model_name] = AutoProcessor.from_pretrained(path)
        elif model_name == 'vit-gpt2':
            self.models[model_name] = VisionEncoderDecoderModel.from_pretrained(path)
            self.processors[model_name] = {
                'feature_extractor': ViTImageProcessor.from_pretrained(path),
                'tokenizer': AutoTokenizer.from_pretrained(path),
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
            }
            self.models[model_name].to(self.processors[model_name]['device'])
        elif model_name == 'blip':
            self.models[model_name] = BlipForConditionalGeneration.from_pretrained(path)
            self.processors[model_name] = BlipProcessor.from_pretrained(path)

    def process_image(self, model_name: str, image_url: str) -> Tuple[str, str]:
        try:
            image = self._load_image(image_url)
            if model_name == 'kosmos-2':
                return self._process_kosmos(image)
            elif model_name == 'vit-gpt2':
                return self._process_vit(image)
            elif model_name == 'blip':
                return self._process_blip(image)
            raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            return 'error', str(e)

    @staticmethod
    def _load_image(url: str) -> Image:
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')

    def _process_kosmos(self, image: Image) -> Tuple[str, str]:
        prompt = "<grounding>An image of"
        inputs = self.processors['kosmos-2'](text=prompt, images=image, return_tensors="pt")
        generated_ids = self.models['kosmos-2'].generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            max_new_tokens=128,
        )
        generated_text = self.processors['kosmos-2'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, _ = self.processors['kosmos-2'].post_process_generation(generated_text)
        return 'ok', processed_text

    def _process_vit(self, image: Image) -> Tuple[str, str]:
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        def predict_step(img: Image):
            final_image = img
            if image.mode != "RGB":
                final_image = image.convert(mode="RGB")

            processor = self.processors["vit-gpt2"]
            pixel_values = processor["feature_extractor"](images=[final_image],
                                                          return_tensors="pt").pixel_values
            device = processor["device"]
            pixel_values = pixel_values.to(device)

            output_ids = self.models["vit-gpt2"].generate(pixel_values, **gen_kwargs)

            preds = processor["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            return preds

        processed_text = predict_step(image)

        return "ok", processed_text[0]

    def _process_blip(self, image: Image) -> Tuple[str, str]:
        inputs = self.processors['blip'](images=image, return_tensors="pt")
        out = self.models['blip'].generate(**inputs)
        processed_text = self.processors['blip'].decode(out[0], skip_special_tokens=True)
        return 'ok', processed_text


app = Flask(__name__)
model_manager = ModelManager()


def create_response(data: Any, status_code: int = HTTPStatus.OK) -> Tuple[Response, int]:
    return jsonify(data), status_code


@app.route('/api/v1/vision/caption', methods=['POST', 'GET'])
def default_process_image_caption() -> Tuple[Response, int]:
    return process_image_caption("kosmos-2")


@app.route('/api/v1/vision/caption/<model_name>', methods=['POST', 'GET'])
def process_image_caption(model_name: str) -> Tuple[Response, int]:
    try:
        data = request.get_json() if request.is_json else request.args
        if not data.get('url'):
            return create_response({'error': 'URL is required'}, HTTPStatus.BAD_REQUEST)

        status, result = model_manager.process_image(model_name, data['url'])
        if status == 'ok':
            response_data = {
                'id': data.get('id', str(uuid.uuid4())),
                'result': {'caption': {'text': result}},
                'model': {
                    'name': model_name,
                    'version': MODEL_CONFIG['MODELS'].get(model_name, {}).get('version', 'latest')
                }
            }
            return create_response(response_data, HTTPStatus.OK)
        return create_response({'error': result}, HTTPStatus.INTERNAL_SERVER_ERROR)
    except Exception as e:
        return create_response({'error': str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
