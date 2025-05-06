PhotoPrism® Computer Vision API
===============================

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-454377.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://img.shields.io/badge/read-the%20docs-4d6a91.svg)](https://docs.photoprism.app/developer-guide/)
[![Community Chat](https://img.shields.io/badge/chat-on%20gitter-4d6a91.svg)](https://link.photoprism.app/chat)
[![GitHub Discussions](https://img.shields.io/badge/ask-%20on%20github-4d6a91.svg)](https://link.photoprism.app/discussions)
[![Bluesky Social](https://dl.photoprism.app/img/badges/badge-bluesky.svg)](https://bsky.app/profile/photoprism.app)
[![Mastodon](https://dl.photoprism.app/img/badges/badge-floss-social.svg)](https://floss.social/@photoprism)

This repository [provides web services](#usage) with advanced [computer vision models](#models) that can be used with [PhotoPrism](https://github.com/photoprism/photoprism) and other applications.

## Table of Contents
<!-- TOC -->
* [PhotoPrism® Computer Vision API](#photoprism-computer-vision-api)
  * [Table of Contents](#table-of-contents)
  * [Local Models](#local-models)
    * [Kosmos-2](#kosmos-2)
    * [VIT-GPT2](#vit-gpt2)
    * [BLIP](#blip)
  * [Remote integrations](#remote-integrations)
    * [OLLAMA](#ollama)
      * [Configuration](#configuration)
    * [Models](#models)
  * [Dependencies](#dependencies)
    * [Flask](#flask)
    * [PyTorch](#pytorch)
    * [Transformers](#transformers)
    * [Pillow](#pillow)
    * [pydantic](#pydantic)
    * [ollama](#ollama-1)
    * [timm](#timm)
    * [huggingface_hub[hf_xet]](#huggingface_hubhf_xet)
  * [Build Setup](#build-setup)
  * [Usage](#usage)
    * [API Endpoints](#api-endpoints)
      * [`/api/v1/vision/caption`](#apiv1visioncaption)
      * [`/api/v1/vision/caption/<model_name>`](#apiv1visioncaptionmodel_name)
      * [`/api/v1/vision/labels/<model_name>`](#apiv1visionlabelsmodel_name)
      * [`/api/v1/vision/nsfw/<model_name>`](#apiv1visionnsfwmodel_name)
    * [Example Request](#example-request)
    * [Example Response](#example-response)
  * [Code Structure](#code-structure)
    * [Internal API](#internal-api)
    * [Model Loading and Initialization](#model-loading-and-initialization)
  * [Request Handlers](#request-handlers)
    * [Default Endpoint](#default-endpoint)
    * [Specific Endpoints](#specific-endpoints)
  * [Contributors](#contributors)
  * [Submitting Pull Requests](#submitting-pull-requests)
  * [License and Disclaimer](#license-and-disclaimer)
<!-- TOC -->

## Local Models

The currently integrated models, each with [its own endpoint](#api-endpoints), are [kosmos-2](#kosmos-2), [vit-gpt2-image-captioning](#vit-gpt2), and [blip-image-captioning large](#blip):

### Kosmos-2

Komsos-2 is the most accurate model of the three. It was developed by Microsoft, and this application uses the transformers implementation of the original model, as described in its [Huggingface](https://huggingface.co/microsoft/kosmos-2-patch14-224). This model was released in June 2023, and offers object detection and spatial reasoning. Kosmos-2 has very accurate image captions (a .04-.1 increase in clip score when compared to the other two models offered), and is the default model used.

### VIT-GPT2

This model was released by [nlpconnect](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning). This model combined VIT and GPT-2 to create a multi-modal image captioning model. I have found this to be the least performing of the three, but your mileage may vary.

### BLIP

This model was released by [Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-large) in 2022. The primary purpose for this model was to increase both image understanding and text generation using novel techniques. It has achieved a +2.8% CIDEr result, and I've found this model to be more performant than VIT-GPT2, but Kosmos-2 to be slightly better (a .4 increase in CLIP score).

### nsfw_image_detector

This model was released by [Freepik](https://huggingface.co/Freepik/nsfw_image_detector). This model can only calculate NSFW weights within four categories: neutral, low, medium, high.

Mapping is done with the best effort to the current API structure.

## Remote integrations

### OLLAMA

Currently, there is implemented ollama integration.

#### Configuration

Ollama usage can be configured through environment variables.

| ENV                   | Default value          | Meaning                             |
|-----------------------|------------------------|-------------------------------------|
| OLLAMA_ENABLED        | false                  | true enables loading of integration |
| OLLAMA_HOST           | http://localhost:11434 | Url to OLLAMA instance              |
| OLLAMA_NSFW_PROMPT    | see code               | Prompt used for NSFW detection      |
| OLLAMA_LABELS_PROMPT  | see code               | Prompt used for label extraction    |
| OLLAMA_CAPTION_PROMPT | see code               | Prompt used for caption extraction  |

### Models

For usage of models in ollama see a [model library](https://ollama.com/library) and official [documentation](https://github.com/ollama/ollama)

Usually you pull model in advance to be available for inference. You can list them with command ollama list. Name of model is in first column including tag.

```aiignore
llava-phi3:latest                             c7edd7b87593    2.9 GB    45 hours ago    
gemma3:4b-it-qat                              d01ad0579247    4.0 GB    45 hours ago    
gemma3:12b-it-qat                             5d4fa005e7bb    8.9 GB    2 days ago      
gemma3:27b-it-qat                             29eb0b9aeda3    18 GB     2 days ago      
gemma3:latest                                 c0494fe00251    3.3 GB    6 weeks ago     
phi4:latest                                   ac896e5b8b34    9.1 GB    2 months ago    
qwen2.5:latest                                845dbda0ea48    4.7 GB    2 months ago 
```

Requirements for running LLM may be roughly estimated from its size. If model has 4 GiB, Then it will probably fit into any GPU with 8 GiB VRAM.
If the model doesn't fit into VRAM, it will run on CPU and it will be much slower (but may be still usable).

Real requirements depend on context length and many other parameters, so you should test manually what model fits your requirements based on the quality of inference and speed of inference on your HW.

## Dependencies

### Flask

[Flask](https://flask.palletsprojects.com/en/3.0.x/) is the framework that is used for the API. It allows for API creation with Python, which is key for this application as it utilizes ML.

### PyTorch

[PyTorch](https://pytorch.org/) is key for working with the ML models to generate the outputs. It also enables GPU processing, speeding up the image processing with the models. PyTorch primarily creates and handles tensors, which are crucial for the function of the models.

### Transformers

[Transformers](https://huggingface.co/docs/transformers/en/index) is used for downloading and loading the models. In addition to this it is used in the image processing with the models.

### Pillow

[Pillow](https://pypi.org/project/pillow/) is used to take the supplied URL and convert it into the format needed to input into the models.

### pydantic

[pydantic](https://github.com/pydantic/pydantic) is used for JSON schemas, serialization and deserialization of requests and responses.

### ollama

[ollama](https://github.com/ollama/ollama) is used as integration library that connects to any given ollama instance.

### timm

[timm](https://huggingface.co/timm) is a tensorflow extension for timm models. Currently used for NSFW detection.

### huggingface_hub[hf_xet]

[xet](https://huggingface.co/blog/xet-on-the-hub) Extension used for faster downloading of huggingface models.

## Build Setup

Before installing the Python dependencies, please make sure that you have [Git](https://git-scm.com/downloads) and [Python 3.12+ (incl. pip)](https://www.python.org/downloads/) installed on your system, e.g. by running the following command on Ubuntu/Debian Linux:

```
sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
```

You can then install the required libraries in a virtual environment by either using the Makefiles we provide (i.e. run `make` in the main project directory or a subdirectory) or by manually running the following commands in a service directory, for example:

```bash
git clone git@github.com:photoprism/photoprism-vision.git
cd photoprism-vision/describe
python3 -m venv ./venv
. ./venv/bin/activate
./venv/bin/pip install --disable-pip-version-check --upgrade pip
./venv/bin/pip install --disable-pip-version-check -r requirements.txt
```

## Usage

Run the Python file `app.py` in the `describe` subdirectory to start the *describe* service after you have installed [the dependencies](#build-setup) (more services, e.g. for OCR and tag generation, may follow):

```bash
./venv/bin/python app.py
```

The service then listens on port 5000 by default and its API endpoints for generating captions support both `GET` and `POST` requests. It can be tested with the `curl` command (`curl.exe` on Windows) as shown in the example below:

```bash
curl -v -H "Content-Type: application/json" \
  --data '{"url":"https://dl.photoprism.app/img/team/avatar.jpg"}' \
  -X POST http://localhost:5000/api/v1/vision/caption
```

At a minimum, a valid image `url` must be specified for this. In addition, a `model` name and an arbitrary `id` [can be passed](#example-request). The API will return the same `id` in [the response](#example-response). If no `id` is passed, a randomly generated UUID will be returned instead.

If your client submits `POST` requests, the request body must be [JSON-encoded](https://www.json.org/), e.g.:

```json
{
    "id": "3487da77-246e-4b4c-9437-67507177bcd7",
    "url": "https://dl.photoprism.app/img/team/avatar.jpg"
}
```

Alternatively, you can perform `GET` requests with URL-encoded query parameters, which is easier to test without an HTTP client:

> http://localhost:5000/api/v1/vision/caption?url=https%3A%2F%2Fdl.photoprism.app%2Fimg%2Fteam%2Favatar.jpg&id=3487da77-246e-4b4c-9437-67507177bcd7

### API Endpoints

#### `/api/v1/vision/caption`

This is the default endpoint of the API. An image url should be passed in with the key "url" or "images" that contains array of base64 encoded images, and optionally a "model" and/or "id" value can be passed in. The "model" key allows the user to specify which of the three models they would like to use. If no model is given, the application will default to using the kosmos-2 model.

#### `/api/v1/vision/caption/<model_name>`

This is the endpoint for a generation of captions. For detailed output see `ApiResponse` and `Caption` classes in api.py

#### `/api/v1/vision/labels/<model_name>`

This is the endpoint for a generation of labels. For detailed output see `ApiResponse` and `Labels` classes in api.py

#### `/api/v1/vision/nsfw/<model_name>`

This is the endpoint for a generation of labels. For detailed output see `ApiResponse` and `NSFW` classes in api.py

### Example Request

`POST /api/v1/vision/caption`

```json
{
    "id": "b0db2187-7a09-438c-8649-a9c6c0f7b8a1",
    "model": "kosmos-2"
    "url": "https://dl.photoprism.app/img/team/avatar.jpg",
}
```

### Example Response

```json
{
    "id": "b0db2187-7a09-438c-8649-a9c6c0f7b8a1",
    "model": {
        "name": "kosmos-2",
        "version": "patch14-224"
    },
    "result": {
        "caption": "An image of a man in a suit smiling."
    }
}
```

## Code Structure

### Internal API

There is predefined internal API in file `api.py`. The class `ImageProcessor` defines methods that any model should provide.

### Model Loading and Initialization

Local models should extend `TorchImageProcessor` class that defines essential abstract methods required to be implemented.

`_get_model_config` returns dictionary with configuration keys `path` = path to the saved model, `source` = huggingface model name, `version` = tag of model.

Usually latest model is downloaded.

`_download_model` downloads specific model and persist it into `models` directory.
`_get_model_name` returns name of model that will be used for selection based on request data
`_load_model` loads chosen model into memory where it will stay until restart

## Request Handlers

Defined in `app.py`

### Default Endpoint

```python
@app.route('/api/v1/vision/caption', methods=['POST', 'GET'])
```

This is the default endpoint. It checks to see if a model is specified, and if it is it calls the service associated with that model and returns the respose with the data. If a model isn't specified it uses kosmos-2.

### Specific Endpoints

```python
@app.route('/api/v1/vision/labels/<model_name>', methods=['POST', 'GET'])
```

There is the endpoint that dynamically routes request to `model_name` in url path variable.

## Contributors

We would like to thank everyone involved, especially [Aatif Dawawala](https://github.com/Aatif-Dawawala) who got things rolling and contributed much of the initial code:

- [Aatif Dawawala](https://github.com/Aatif-Dawawala)
- [Niaz Faridani-Rad](https://github.com/derneuere)

[Learn more ›](https://github.com/photoprism/photoprism-vision/graphs/contributors)
 
## Submitting Pull Requests

Follow our [step-by-step guide](https://docs.photoprism.app/developer-guide/pull-requests) to learn how to submit new features, bug fixes, and documentation enhancements.

[Learn more ›](https://docs.photoprism.app/developer-guide/pull-requests)

## License and Disclaimer

The files in this repository are licensed under the [Apache License, Version 2.0](https://docs.photoprism.app/license/apache/) (the “License”).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

[Learn more ›](https://docs.photoprism.app/license/apache/)

----

*Copyright © 2024 [PhotoPrism UG](https://www.photoprism.app/contact). By using the software and services we provide, you agree to our [Terms of Service](https://www.photoprism.app/terms), [Privacy Policy](https://www.photoprism.app/privacy), and [Code of Conduct](https://www.photoprism.app/code-of-conduct). PhotoPrism® is a [registered trademark](https://www.photoprism.app/trademark).*
