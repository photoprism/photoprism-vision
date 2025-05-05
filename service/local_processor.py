import os
from typing import Tuple, Dict, Any

import torch
from PIL.Image import Image as ImageType, Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BlipForConditionalGeneration, \
    BlipProcessor, TimmWrapperForImageClassification, ViTImageProcessor, \
    VisionEncoderDecoderModel
from typing_extensions import override

from processor import ImageProcessor
from api import Labels, NSFW, NSFWProbabilities

# Configuration Constants
MODEL_CONFIG = {
    'BASE_DIR': 'models',
    'MODELS': {
        'kosmos-2': {
            'path': 'models/kosmos-2-patch14-224',
            'source': 'microsoft/kosmos-2-patch14-224',
            'version': 'patch14-224',
        },
        'vit-gpt2': {
            'path': 'models/vit-gpt2-image-captioning',
            'source': 'nlpconnect/vit-gpt2-image-captioning',
            'version': 'latest',
        },
        'blip': {
            'path': 'models/blip-image-captioning-large',
            'source': 'Salesforce/blip-image-captioning-large',
            'version': 'latest',
        },
        'nsfw': {
            'path': 'models/nsfw_image_detector',
            'source': 'Freepik/nsfw_image_detector',
            'version': 'latest',
        }
    }
}


class LocalImageProcessor(ImageProcessor):
    def __init__(self, download_all_at_startup=True):
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self._ensure_model_dirs()

        # Download all models at first start if requested
        if download_all_at_startup:
            self._download_all_models()

    def _ensure_model_dirs(self):
        """Ensure model directories exist without loading the models."""
        os.makedirs(MODEL_CONFIG['BASE_DIR'], exist_ok=True)
        # Create model directories but don't load the models yet
        for model_name, config in MODEL_CONFIG['MODELS'].items():
            os.makedirs(config['path'], exist_ok=True)

    def _download_all_models(self):
        """Download all models at first start."""
        print("Downloading all models...")
        for model_name, config in MODEL_CONFIG['MODELS'].items():
            self._download_model_if_needed(model_name)
        print("All models downloaded successfully.")

    def _download_model_if_needed(self, model_name: str) -> str:
        """Download the model if it doesn't exist and return the path."""
        config = MODEL_CONFIG['MODELS'].get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")

        path = config['path']
        source = config['source']

        # Check if model is already downloaded
        if not os.path.exists(os.path.join(path, "config.json")):
            print(f"Downloading {source}...")
            self._download_model(source, path)
            print(f"Downloaded {source} to {path}")

        return path

    @staticmethod
    def _download_model(source: str, path: str):
        """Download a model from the source to the specified path."""
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
        elif 'nsfw' in source:
            TimmWrapperForImageClassification.from_pretrained(source).save_pretrained(path)
            AutoProcessor.from_pretrained(source).save_pretrained(path)
        else:
            raise ValueError(f"Unknown model source: {source}")

    def _load_model_if_needed(self, model_name: str):
        """Lazy-load a model only when it's needed."""
        if model_name in self.models and model_name in self.processors:
            return

        path = self._download_model_if_needed(model_name)

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
        elif model_name == 'nsfw':
            self.models[model_name] = TimmWrapperForImageClassification.from_pretrained(path)
            self.processors[model_name] = AutoProcessor.from_pretrained(path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Loaded model: {model_name}")

    def can_process(self, model_name: str) -> bool:
        return model_name in MODEL_CONFIG['MODELS']

    @override
    def generate_caption(self, model_name: str, image: ImageType) -> Tuple[str, str]:
        try:
            self._load_model_if_needed(model_name)

            if model_name == 'kosmos-2':
                return self._process_kosmos(image)
            elif model_name == 'vit-gpt2':
                return self._process_vit(image)
            elif model_name == 'blip':
                return self._process_blip(image)
            raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            return 'error', str(e)

    @override
    def generate_labels(self, model_name: str, image: Image) -> Tuple[str, Labels | str]:
        # TODO: Implement labels generation
        pass

    @override
    def detect_nsfw(self, model_name: str, image: Image) -> Tuple[str, NSFW | str]:
        try:
            self._load_model_if_needed('nsfw')

            model = self.models['nsfw']
            processor = self.processors['nsfw']

            # Process the image
            inputs = processor(images=image, return_tensors="pt")

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                neutral, low, medium, high = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

            nsfw_probs = NSFWProbabilities(
                Neutral=neutral,
                Drawing=0.0,
                Hentai=medium,
                Porn=high,
                Sexy=low
            )

            return 'ok', NSFW(nsfw=[nsfw_probs])
        except Exception as e:
            return 'error', str(e)

    def _process_kosmos(self, image: ImageType) -> Tuple[str, str]:
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

    def _process_vit(self, image: ImageType) -> Tuple[str, str]:
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        def predict_step():
            final_image = image
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

        processed_text = predict_step()

        return "ok", processed_text[0]

    def _process_blip(self, image: ImageType) -> Tuple[str, str]:
        inputs = self.processors['blip'](images=image, return_tensors="pt")
        out = self.models['blip'].generate(**inputs)
        processed_text = self.processors['blip'].decode(out[0], skip_special_tokens=True)
        return 'ok', processed_text
