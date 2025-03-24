import os
import sys
import logging
from flask import Flask, jsonify, request
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, VisionEncoderDecoderModel, ViTImageProcessor, \
    AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define model paths
MODEL_DIR = "models"
KOSMOS_MODEL_PATH = os.path.join(MODEL_DIR, "kosmos-2-patch14-224")
VIT_MODEL_PATH = os.path.join(MODEL_DIR, "vit-gpt2-image-captioning")
BLIP_MODEL_PATH = os.path.join(MODEL_DIR, "blip-image-captioning-large")

# Global state management
class ModelManager:
    def __init__(self):
        self.device = None
        self.kosmos = {'model': None, 'processor': None}
        self.vit = {'model': None, 'processor': None, 'tokenizer': None}
        self.blip = {'model': None, 'processor': None}
        self.initialize_gpu()

    def initialize_gpu(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                # Set memory limits
                torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                self.device = torch.device("cpu")
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"GPU initialization error: {e}")
            self.device = torch.device("cpu")

    def cleanup(self):
        try:
            # Clear model memory
            self.kosmos = {'model': None, 'processor': None}
            self.vit = {'model': None, 'processor': None, 'tokenizer': None}
            self.blip = {'model': None, 'processor': None}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

model_manager = ModelManager()

def download_model(model_name, save_path):
    try:
        if not os.path.exists(save_path):
            print(f"Downloading {model_name}...")
            if model_name == "microsoft/kosmos-2-patch14-224":
                AutoModelForVision2Seq.from_pretrained(model_name).save_pretrained(save_path)
                AutoProcessor.from_pretrained(model_name).save_pretrained(save_path)
            elif model_name == "nlpconnect/vit-gpt2-image-captioning":
                VisionEncoderDecoderModel.from_pretrained(model_name).save_pretrained(save_path)
                ViTImageProcessor.from_pretrained(model_name).save_pretrained(save_path)
                AutoTokenizer.from_pretrained(model_name).save_pretrained(save_path)
            elif model_name == "Salesforce/blip-image-captioning-large":
                BlipForConditionalGeneration.from_pretrained(model_name).save_pretrained(save_path)
                BlipProcessor.from_pretrained(model_name).save_pretrained(save_path)
            print(f"{model_name} downloaded and saved to {save_path}")
        else:
            print(f"{model_name} already exists at {save_path}")
    except Exception as e:
        logger.error(f"Model download error for {model_name}: {e}")
        raise

# Download models
os.makedirs(MODEL_DIR, exist_ok=True)
download_model("microsoft/kosmos-2-patch14-224", KOSMOS_MODEL_PATH)
download_model("nlpconnect/vit-gpt2-image-captioning", VIT_MODEL_PATH)
download_model("Salesforce/blip-image-captioning-large", BLIP_MODEL_PATH)

@app.before_first_request
def initialize():
    """Initialize on first request"""
    try:
        # Load Kosmos model
        model_manager.kosmos['model'] = AutoModelForVision2Seq.from_pretrained(KOSMOS_MODEL_PATH)
        model_manager.kosmos['processor'] = AutoProcessor.from_pretrained(KOSMOS_MODEL_PATH)
        model_manager.kosmos['model'].to(model_manager.device)
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        raise

def kosmosGenerateResponse(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        return "fetchError", f"Unable to fetch image: {str(e)}"

    prompt = "<grounding>An image of"

    try:
        inputs = model_manager.kosmos['processor'](text=prompt, images=image, return_tensors="pt")
        generated_ids = model_manager.kosmos['model'].generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )

        generated_text = kosmosProcessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, entities = kosmosProcessor.post_process_generation(generated_text)
    except Exception as e:
        logger.error(f"Kosmos processing error: {e}")
        return "processingError", str(e)

    return "ok", processed_text

def vitGenerateResponse(url):
    global VITLoaded
    global vitModel, vitFeature_extractor, vitTokenizer, device
    if not VITLoaded:
        vitModel = VisionEncoderDecoderModel.from_pretrained(VIT_MODEL_PATH)
        vitFeature_extractor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
        vitTokenizer = AutoTokenizer.from_pretrained(VIT_MODEL_PATH)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vitModel.to(device)
        VITLoaded = True

    vitModel.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(url):
        image = Image.open(requests.get(url, stream=True).raw)
        images = []

        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images.append(image)
        pixel_values = vitFeature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = vitModel.generate(pixel_values, **gen_kwargs)
        preds = vitTokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    processed_text = predict_step(url)  # returns prediction
    return "ok", processed_text

@app.route('/api/v1/vision/describe', methods=['POST', 'GET'])
def generateResponse():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
    elif request.method == 'GET':
        data = request.args

    url = data.get('url')
    model = data.get('model')
    id = data.get('id')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if model == "kosmos-2" or not model:
        status, result = kosmosGenerateResponse(url)
        if status == "fetchError":
            return jsonify({"error": result}), 500
        elif status == "processingError":
            return jsonify({"error": result}), 500
        elif status == "ok":
            if id:
                return jsonify({"id": id, "result": {"caption": result}, "model": {"name": "kosmos-2", "version": "patch14-224"}}), 200
            return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": "kosmos-2", "version": "patch14-224"}}), 200

    return jsonify({"error": "Error during processing"})

@app.route('/api/v1/vision/describe/vit-gpt2-image-captioning', methods=['POST', 'GET'])
def vitController():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
    elif request.method == 'GET':
        data = request.args

    url = data.get('url')
    id = data.get('id')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    status, result = vitGenerateResponse(url)

    if status == "ok":
        if id:
            return jsonify({"id": id, "result": {"caption": result}, "model": {"name": "vit-gpt2-image-captioning", "version": "latest"}}), 200
        return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": "vit-gpt2-image-captioning", "version": "latest"}}), 200

    return jsonify({"error": "Error during processing"})

@app.route('/api/v1/vision/describe/blip-image-captioning-large', methods=['POST', 'GET'])
def blipController():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
    elif request.method == 'GET':
        data = request.args

    url = data.get('url')
    id = data.get('id')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    status, result = blipGenerateResponse(url)

    if status == "ok":
        if id:
            return jsonify({"id": id, "result": {"caption": result}, "model": {"name": "blip-image-captioning-large", "version": "latest"}}), 200
        return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": "blip-image-captioning-large", "version": "latest"}}), 200

    return jsonify({"error": "Error during processing"})

# Add health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gpu': torch.cuda.is_available(),
        'models': {
            'kosmos': model_manager.kosmos['model'] is not None,
            'vit': model_manager.vit['model'] is not None,
            'blip': model_manager.blip['model'] is not None
        }
    })

if __name__ == '__main__':
    # Production settings
    app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)
