import logging
import os
from typing import Tuple, Dict
from abc import ABC, abstractmethod

import torch
from PIL.Image import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    TimmWrapperForImageClassification,
    ViTImageProcessor,
    VisionEncoderDecoderModel
)
from typing_extensions import override

from api import Labels, NSFW, NSFWProbabilities
from processor import ImageProcessor

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
        'nsfw_image_detector': {
            'path': 'models/nsfw_image_detector',
            'source': 'Freepik/nsfw_image_detector',
            'version': 'latest',
        }
    }
}

logger = logging.getLogger(__name__)


class TorchImageProcessor(ABC):
    def __init__(self):
        self.model = None
        self.processor = None
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        config = self._get_model_config()
        os.makedirs(MODEL_CONFIG['BASE_DIR'], exist_ok=True)
        os.makedirs(config['path'], exist_ok=True)

    def download_model_if_needed(self):
        """Download the model if it doesn't exist and return the path."""
        config = self._get_model_config()
        path = config['path']
        source = config['source']

        # Check if the model is already downloaded
        if not os.path.exists(os.path.join(path, "config.json")):
            logger.info(f"Downloading {source}...")
            self._download_model(source, path)
            logger.info(f"Downloaded {source} to {path}")

    @abstractmethod
    def _get_model_config(self) -> Dict[str, str]:
        """Return the model configuration."""
        pass

    @abstractmethod
    def _load_model(self):
        """Load the model and processor."""
        pass

    @abstractmethod
    def _download_model(self, source: str, path: str):
        """Download a model from the source to the specified path."""
        pass

    def load_if_needed(self):
        """Lazy-load the model only when it's necessary."""
        if self.model is None or self.processor is None:
            self.download_model_if_needed()
            self._load_model()
            logger.info(f"Loaded model: {self._get_model_name()}")

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def generate_caption(self, image: Image) -> Tuple[str, str]:
        pass


class Kosmos2Processor(TorchImageProcessor):
    """Processor for the Kosmos-2 model."""

    @override
    def _get_model_config(self) -> Dict[str, str]:
        return MODEL_CONFIG['MODELS'][self._get_model_name()]

    @override
    def _get_model_name(self) -> str:
        return 'kosmos-2'

    @override
    def _download_model(self, source: str, path: str):
        AutoModelForVision2Seq.from_pretrained(source).save_pretrained(path)
        AutoProcessor.from_pretrained(source).save_pretrained(path)

    @override
    def _load_model(self):
        path = self._get_model_config()['path']
        self.model = AutoModelForVision2Seq.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(path)

    @override
    def generate_caption(self, image: Image) -> Tuple[str, str]:
        try:
            self.load_if_needed()

            prompt = "<grounding>An image of"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                max_new_tokens=128,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text, _ = self.processor.post_process_generation(generated_text)
            return 'ok', processed_text
        except Exception as e:
            return 'error', str(e)


class VitGpt2Processor(TorchImageProcessor):
    """Processor for the ViT-GPT2 model."""

    @override
    def _get_model_config(self) -> Dict[str, str]:
        return MODEL_CONFIG['MODELS'][self._get_model_name()]

    @override
    def _get_model_name(self) -> str:
        return 'vit-gpt2'

    @override
    def _download_model(self, source: str, path: str):
        VisionEncoderDecoderModel.from_pretrained(source).save_pretrained(path)
        ViTImageProcessor.from_pretrained(source).save_pretrained(path)
        AutoTokenizer.from_pretrained(source).save_pretrained(path)

    @override
    def _load_model(self):
        path = self._get_model_config()['path']
        self.model = VisionEncoderDecoderModel.from_pretrained(path)
        self.processor = {
            'feature_extractor': ViTImageProcessor.from_pretrained(path),
            'tokenizer': AutoTokenizer.from_pretrained(path),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        self.model.to(self.processor['device'])

    @override
    def generate_caption(self, image: Image) -> Tuple[str, str]:
        try:
            self.load_if_needed()

            max_length = 16
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

            final_image = image
            if image.mode != "RGB":
                final_image = image.convert(mode="RGB")

            pixel_values = self.processor["feature_extractor"](
                images=[final_image],
                return_tensors="pt"
            ).pixel_values

            device = self.processor["device"]
            pixel_values = pixel_values.to(device)

            output_ids = self.model.generate(pixel_values, **gen_kwargs)

            preds = self.processor["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            return "ok", preds[0]
        except Exception as e:
            return 'error', str(e)


class BlipImageProcessor(TorchImageProcessor):
    """Processor for the BLIP model."""

    @override
    def _get_model_config(self) -> Dict[str, str]:
        return MODEL_CONFIG['MODELS'][self._get_model_name()]

    @override
    def _get_model_name(self) -> str:
        return 'blip'

    @override
    def _download_model(self, source: str, path: str):
        BlipForConditionalGeneration.from_pretrained(source).save_pretrained(path)
        BlipProcessor.from_pretrained(source).save_pretrained(path)

    @override
    def _load_model(self):
        path = self._get_model_config()['path']
        self.model = BlipForConditionalGeneration.from_pretrained(path)
        self.processor = BlipProcessor.from_pretrained(path)

    @override
    def generate_caption(self, image: Image) -> Tuple[str, str]:
        try:
            self.load_if_needed()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = self.model.generate(**inputs)
            processed_text = self.processor.decode(out[0], skip_special_tokens=True)

            return 'ok', processed_text
        except Exception as e:
            return 'error', str(e)


class NSFWImageProcessor(TorchImageProcessor):
    """Processor for NSFW image detection."""

    @override
    def _get_model_config(self) -> Dict[str, str]:
        return MODEL_CONFIG['MODELS'][self._get_model_name()]

    @override
    def _get_model_name(self) -> str:
        return 'nsfw_image_detector'

    @override
    def _download_model(self, source: str, path: str):
        TimmWrapperForImageClassification.from_pretrained(source).save_pretrained(path)
        AutoProcessor.from_pretrained(source).save_pretrained(path)

    @override
    def _load_model(self):
        path = self._get_model_config()['path']
        self.model = TimmWrapperForImageClassification.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(path)

    @override
    def generate_caption(self, image: Image) -> Tuple[str, str]:
        return 'error', "This model does not support caption generation"

    def detect_nsfw(self, image: Image) -> Tuple[str, NSFW | str]:
        try:
            self.load_if_needed()

            # Process the image
            inputs = self.processor(images=image, return_tensors="pt")

            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
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


class ProcessorFactory:
    """Factory for creating processor instances."""

    @staticmethod
    def create_processor(model_name: str) -> TorchImageProcessor:
        if model_name == 'kosmos-2':
            return Kosmos2Processor()
        elif model_name == 'vit-gpt2':
            return VitGpt2Processor()
        elif model_name == 'blip':
            return BlipImageProcessor()
        elif model_name == 'nsfw_image_detector':
            return NSFWImageProcessor()
        else:
            raise ValueError(f"Unknown model: {model_name}")


class LocalImageProcessor(ImageProcessor):
    """Manager class that coordinates local image processors."""

    def __init__(self, download_all_at_startup=True):
        self.processors = {}
        self._ensure_model_dirs()

        # Download all models at first start if requested
        if download_all_at_startup:
            self._download_all_models()

    @staticmethod
    def _ensure_model_dirs():
        """Ensure model directories exist without loading the models."""
        os.makedirs(MODEL_CONFIG['BASE_DIR'], exist_ok=True)
        # Create model directories but don't load the models yet
        for model_name, config in MODEL_CONFIG['MODELS'].items():
            os.makedirs(config['path'], exist_ok=True)

    @staticmethod
    def _download_all_models():
        """Download all models at the first start."""
        logger.info("Downloading all models...")
        for model_name in MODEL_CONFIG['MODELS']:
            processor = ProcessorFactory.create_processor(model_name)
            processor.download_model_if_needed()
        logger.info("All models downloaded successfully.")

    def get_processor(self, model_name: str) -> TorchImageProcessor:
        """Get or create a processor for the specified model."""
        if model_name not in self.processors:
            self.processors[model_name] = ProcessorFactory.create_processor(model_name)
        return self.processors[model_name]

    @override
    def can_process(self, model_name: str) -> bool:
        """Check if the specified model is supported."""
        return model_name in MODEL_CONFIG['MODELS']

    @override
    def generate_caption(self, model_name: str, image: Image) -> Tuple[str, str]:
        processor = self.get_processor(model_name)
        return processor.generate_caption(image)

    @override
    def generate_labels(self, model_name: str, images: list[Image]) -> Tuple[str, Labels | str]:
        # TODO: Implement label generation for local models
        return 'error', 'Local model does not support label generation yet. Use the Ollama API instead.'

    @override
    def detect_nsfw(self, model_name: str, image: Image) -> Tuple[str, NSFW | str]:
        """Detect NSFW content in the image using the specified model."""
        processor = self.get_processor(model_name)
        if isinstance(processor, NSFWImageProcessor):
            return processor.detect_nsfw(image)
        raise ValueError(f"Model {model_name} does not support NSFW detection")
