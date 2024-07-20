from flask import Flask, jsonify, request
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and processor at startup
kosmosModel = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
kosmosProcessor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

vitModel = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vitFeature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vitTokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.route('/api/v1/vision/describe/kosmos-2/patch14-224', methods=['POST'])
def kosmosGenerateResponse():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        return jsonify({"error": f"Unable to fetch image: {str(e)}"}), 500

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
        return jsonify({"error": f"Error during processing: {str(e)}"}), 500

    return jsonify({"processed_text": processed_text}), 200

@app.route('/api/v1/vision/describe/vit-gpt2-image-captioning', methods=['POST'])
def vitGenerateResponse():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    

    vitModel.to(device)

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

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

    processed_text = predict_step(url) # returns prediction

    return jsonify({"processed_text": processed_text}), 200

if __name__ == '__main__':
    app.run(debug=True)
