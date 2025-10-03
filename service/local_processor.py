from pyexpat import model
import logging, traceback
import os, yaml, gettext, threading
from abc import ABC, abstractmethod

import timm
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
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
    CLIPModel,
    CLIPProcessor
)
from typing_extensions import override

from api import Labels, Label, NSFW, NSFWProbabilities
from processor import ImageProcessor
from ai.classify.labels_utils import LabelRulesProcessor

# set language
lang_code = os.getenv("LABELS_LOCALE", "en")  # Default "en"
translation = gettext.translation('rules_locale', localedir='/app/locales', languages=[lang_code], fallback=True)
translation.install()

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
        },
        'efficientnet_b0.ra_in1k': {
            'path': 'models/efficientnet_b0.ra_in1k',
            'source': 'timm/efficientnet_b0.ra_in1k',
            'version': 'latest',
        },
        'efficientvit_l3.r384_in1k': {
            'path': 'models/efficientvit_l3.r384_in1k',
            'source': 'timm/efficientvit_l3.r384_in1k',
            'version': 'latest',
        },
        'tf_efficientnetv2_l.in1k': {
            'path': 'models/tf_efficientnetv2_l.in1k',
            'source': 'timm/tf_efficientnetv2_l.in1k',
            'version': 'latest',
        },
        'convnextv2_huge.fcmae_ft_in22k_in1k_384': {
            'path': 'models/convnextv2_huge.fcmae_ft_in22k_in1k_384',
            'source': 'timm/convnextv2_huge.fcmae_ft_in22k_in1k_384',
            'version': 'latest',
        },
        'convnextv2_huge.fcmae_ft_in22k_in1k_512': {
            'path': 'models/convnextv2_huge.fcmae_ft_in22k_in1k_512',
            'source': 'timm/convnextv2_huge.fcmae_ft_in22k_in1k_512',
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
    def _get_model_config(self) -> dict[str, str]:
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
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
        pass


class Kosmos2Processor(TorchImageProcessor):
    """Processor for the Kosmos-2 model."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__()

    @override
    def _get_model_config(self) -> dict[str, str]:
        return MODEL_CONFIG['MODELS'][self.model_name]

    @override
    def _get_model_name(self) -> str:
        return self.model_name

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
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
        try:
            self.load_if_needed()
            if prompt == '' or prompt == 'default':
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
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__()

    @override
    def _get_model_config(self) -> dict[str, str]:
        return MODEL_CONFIG['MODELS'][self.model_name]

    @override
    def _get_model_name(self) -> str:
        return self.model_name

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
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
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
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__()

    @override
    def _get_model_config(self) -> dict[str, str]:
        return MODEL_CONFIG['MODELS'][self.model_name]

    @override
    def _get_model_name(self) -> str:
        return self.model_name

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
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
        try:
            self.load_if_needed()
            if prompt == "" or prompt == 'default':
                prompt = "an image of"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            inputs = self.processor(image, prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = self.model.generate(**inputs)
            processed_text = self.processor.decode(out[0], skip_special_tokens=True)

            return 'ok', processed_text
        except Exception as e:
            return 'error', str(e)


class NSFWImageProcessor(TorchImageProcessor):
    """Processor for NSFW image detection."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__()

    @override
    def _get_model_config(self) -> dict[str, str]:
        return MODEL_CONFIG['MODELS'][self.model_name]

    @override
    def _get_model_name(self) -> str:
        return self.model_name

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
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
        return 'error', "This model does not support caption generation"

    def detect_nsfw(self, image: Image) -> tuple[str, NSFW | str]:
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

# NOTE: Only supports AutoModelForImageClassification with ImageNet-1K classes.
class HFImageClassificationProcessor(TorchImageProcessor):
    """Processor for huggingface classification models."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        # get labels rules
        rules_path = '/app/assets/classify/rules.yml'
        self._labels_rules = LabelRulesProcessor(rules_path)
        # get imagenet1k labels
        labels_path = "/app/assets/classify/labels_imagenet1K.txt"
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels_1K = [line.strip() for line in f if line.strip()]

        super().__init__()

    @override
    def _get_model_config(self) -> dict[str, str]:
        return MODEL_CONFIG['MODELS'][self.model_name]

    @override
    def _get_model_name(self) -> str:
        return self.model_name

    @override
    def _download_model(self, source: str, path: str):
        try:
            # read config to judge model_type
            config = AutoConfig.from_pretrained(source)
            model_type = config.model_type
            
            if model_type == "clip":
                # CLIP model
                CLIPModel.from_pretrained(source).save_pretrained(path)
                CLIPProcessor.from_pretrained(source).save_pretrained(path)
            else:
                # classification model
                AutoModelForImageClassification.from_pretrained(source).save_pretrained(path)
                AutoImageProcessor.from_pretrained(source)
        except Exception as e:
            logger.error(f"Error generating labels: {e}")

    @override
    def _load_model(self):
        try:
            path = self._get_model_config()['path']

            # read config to judge model_type
            config = AutoConfig.from_pretrained(path)
            model_type = config.model_type

            if model_type == "clip":
                # CLIP model
                self.model = CLIPModel.from_pretrained(path)
                self.processor = CLIPProcessor.from_pretrained(path)
            else:
                # classification model
                self.model = AutoModelForImageClassification.from_pretrained(path)
                self.processor = AutoImageProcessor.from_pretrained(path)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error generating labels: {e}")

    @override
    def generate_caption(self, image: Image, prompt) -> tuple[str, str]:
        return 'error', "This model does not support caption generation"

    def generate_labels(self, images: Image, prompt) -> tuple[str, str]:
        try:
            logger.info(f"Generating labels started for model {self.model_name}")
            self.load_if_needed()
            logger.info(f"load_if_needed done")

            labels = []
            for image in images:
                if self.model.config.model_type == "clip":
                    # CLIP model
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    output = self.model.get_image_features(**inputs)
                else:
                    # ImageClassification model
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    output = self.model(**inputs).logits
                logits = output

                top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1)[0], k=5)

                top_label = self.labels_1K[top5_class_indices[0].item()].lower()
                top_prob = top5_probabilities[0].item()

                trans = self._labels_rules.transform_label(top_label, top_prob)
                if(trans['state']=='fail'):
                    err_msg = f'transform label fail: {top_label} {top_prob:.2f}, please check rules.'
                    logger.warning(err_msg)
                    return 'error', err_msg

                lbl_name = trans['label'].lower()
                categories = trans['categories']
                conf = trans['conf']

                if(categories != None):
                    categories = [_(item.lower()) for item in categories]

                logger.info(f"Image classified as {_(lbl_name)} with confidence {conf:.4f}")
                output_label = Label(
                    name=_(lbl_name),
                    categories = categories,
                    confidence=conf,
                )
                labels.append(output_label)

            results = Labels(labels=labels)
            return 'ok', results
        except Exception as e:
            logger.error(f"Error generating labels: {traceback.format_exc()}")
            return 'error', str(e)

class ProcessorFactory:
    """Factory for creating processor instances."""

    @staticmethod
    def create_processor(model_name: str) -> TorchImageProcessor:
        if model_name == 'kosmos-2':
            return Kosmos2Processor(model_name)
        elif model_name == 'vit-gpt2':
            return VitGpt2Processor(model_name)
        elif model_name == 'blip':
            return BlipImageProcessor(model_name)
        elif model_name == 'nsfw_image_detector':
            return NSFWImageProcessor(model_name)
        elif model_name == 'efficientnet_b0.ra_in1k':
            return HFImageClassificationProcessor(model_name)
        elif model_name == 'efficientvit_l3.r384_in1k':
            return HFImageClassificationProcessor(model_name)
        elif model_name == 'tf_efficientnetv2_l.in1k':
            return HFImageClassificationProcessor(model_name)
        elif model_name == 'convnextv2_huge.fcmae_ft_in22k_in1k_384':
            return HFImageClassificationProcessor(model_name)
        elif model_name == 'convnextv2_huge.fcmae_ft_in22k_in1k_512':
            return HFImageClassificationProcessor(model_name)
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
            logger.info(f"{model_name}")
        logger.info("All models downloaded successfully.")

    def get_processor(self, model_name: str) -> TorchImageProcessor:
        """Get or create a processor for the specified model."""
        if model_name not in self.processors:
            self.processors[model_name] = ProcessorFactory.create_processor(model_name)
        return self.processors[model_name]

    @override
    def can_process(self, model_name: str, model_version: str) -> bool:
        """Check if the specified model is supported."""
        return model_name in MODEL_CONFIG['MODELS']

    @override
    def generate_caption(self, model_name: str, model_version: str, image: Image, prompt) -> tuple[str, str]:
        processor = self.get_processor(model_name)
        return processor.generate_caption(image, prompt)

    @override
    def generate_labels(self, model_name: str, model_version: str, images: list[Image], prompt: str) -> tuple[str, Labels | str]:
        processor = self.get_processor(model_name)
        return processor.generate_labels(images, prompt)

    @override
    def detect_nsfw(self, model_name: str, model_version: str, image: Image, prompt) -> tuple[str, NSFW | str]:
        """Detect NSFW content in the image using the specified model."""
        processor = self.get_processor(model_name)
        if isinstance(processor, NSFWImageProcessor):
            return processor.detect_nsfw(image)
        raise ValueError(f"Model {model_name} does not support NSFW detection")
