import os
from flask import Flask, jsonify, request
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, VisionEncoderDecoderModel, ViTImageProcessor, \
    AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
import uuid

app = Flask(__name__)

# Define model paths
MODEL_DIR = "models"
KOSMOS_MODEL_PATH = os.path.join(MODEL_DIR, "kosmos-2-patch14-224")
VIT_MODEL_PATH = os.path.join(MODEL_DIR, "vit-gpt2-image-captioning")
BLIP_MODEL_PATH = os.path.join(MODEL_DIR, "blip-image-captioning-large")


def download_model(model_name, save_path):
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


# Download models
os.makedirs(MODEL_DIR, exist_ok=True)
download_model("microsoft/kosmos-2-patch14-224", KOSMOS_MODEL_PATH)
download_model("nlpconnect/vit-gpt2-image-captioning", VIT_MODEL_PATH)
download_model("Salesforce/blip-image-captioning-large", BLIP_MODEL_PATH)

# Load models
print("Loading models...")
kosmosModel = AutoModelForVision2Seq.from_pretrained(KOSMOS_MODEL_PATH)
kosmosProcessor = AutoProcessor.from_pretrained(KOSMOS_MODEL_PATH)

vitModel = VisionEncoderDecoderModel.from_pretrained(VIT_MODEL_PATH)
vitFeature_extractor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
vitTokenizer = AutoTokenizer.from_pretrained(VIT_MODEL_PATH)

blipProcessor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
blipModel = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vitModel.to(device)

def kosmosGenerateResponse(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        return "fetchError", f"Unable to fetch image: {str(e)}"

    prompt = "<grounding>An image of"

    try:
        inputs = kosmosProcessor(text=prompt, images=image, return_tensors="pt")
        generated_ids = kosmosModel.generate(
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
        return "processingError", f"Error during processing: {str(e)}"

    return "ok", processed_text

def vitGenerateResponse(url):
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

def blipGenerateResponse(url):
    img_url = url
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = blipProcessor(raw_image, return_tensors="pt")

    out = blipModel.generate(**inputs)
    processed_text = blipProcessor.decode(out[0], skip_special_tokens=True)

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
    elif model == "vit-gpt2-image-captioning":
        status, result = vitGenerateResponse(url)
        if status == "ok":
            if id:
                return jsonify({"id": id, "result": {"caption": result}, "model": {"name": model, "version": "latest"}}), 200
            return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": model, "version": "latest"}}), 200
        return jsonify({"error": "Error during processing"})
    elif model == "blip-image-captioning-large":
        status, result = blipGenerateResponse(url)
        if status =='ok':
            if id:
                return jsonify({"id": id, "result": {"caption": result}, "model": {"name": model, "version": "latest"}}), 200
            return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": model, "version": "latest"}}), 200
        return jsonify({"error": "Error during processing"})        







@app.route('/api/v1/vision/describe/kosmos-2/patch14-224', methods=['POST', 'GET'])
def kosmosController():
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
    
    status, result = kosmosGenerateResponse(url)

    if status == "fetchError":
        return jsonify({"error": result}), 500
    elif status == "processingError":
        return jsonify({"error": result}), 500
    elif status == "ok":
        if id:
            return jsonify({"id": id, "result": {"caption": result}, "model": {"name": "kosmos-2", "version": "patch14-224"}}), 200
        return jsonify({"id": uuid.uuid4(), "result": {"caption": result}, "model": {"name": "kosmos-2", "version": "patch14-224"}}), 200

    


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
    
    return jsonify({"error", "Error during processing"})




if __name__ == '__main__':
    app.run(port=5000, debug=True)
