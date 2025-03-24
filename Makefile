BUILD_GIT ?= $(shell (cd .. && git describe --always))
BUILD_DATE ?= $(shell date -u +%y%m%d)
BUILD_TAG ?= $(BUILD_DATE)-$(BUILD_GIT)

UNAME := $(shell uname)

all: pip install
deps: pip upgrade
install: venv

build: docker-build
docker-build:
	(cd describe && make docker-build)

preview: preview-describe
preview-describe:
	./build.sh describe linux/amd64

release: release-describe
release-describe:
	./build.sh describe linux/amd64,linux/arm64 $(BUILD_DATE)

start:
	docker compose --profile=all pull --ignore-pull-failures
	docker compose up -d
	docker compose logs -f || true
stop:
	docker compose down -v
terminal:
	docker compose exec vision-describe bash
logs:
	docker compose logs -f || true

pip:
ifeq ($(UNAME), Linux)
	sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
endif

venv: describe/venv
describe/venv:
	(cd describe && make venv)

upgrade: upgrade-describe
upgrade-describe:
	(cd describe && make upgrade)

# Container configuration
IMAGE_NAME := photoprism-vision
VERSION := latest
DOCKER_REGISTRY := docker.io/photoprism
PODMAN_REGISTRY := quay.io/photoprism

# Choose container engine (docker or podman)
ENGINE ?= docker

# Common commands
BUILD_CMD = $(ENGINE) build -t $(IMAGE_NAME):$(VERSION) .
RUN_CMD = $(ENGINE) run -p 5000:5000 -v $(PWD)/models:/app/models
PUSH_CMD = $(ENGINE) push

.PHONY: all build run push clean k8s-deploy

all: build

# Build image with either docker or podman
build:
	$(BUILD_CMD)

# Run container with GPU support
run:
	$(RUN_CMD) --gpus all $(IMAGE_NAME):$(VERSION)

# Run without GPU
run-cpu:
	$(RUN_CMD) $(IMAGE_NAME):$(VERSION)

# Push to registry based on engine
push-docker:
	$(BUILD_CMD)
	$(PUSH_CMD) $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)

push-podman:
	$(BUILD_CMD)
	$(PUSH_CMD) $(PODMAN_REGISTRY)/$(IMAGE_NAME):$(VERSION)

# Deploy to kubernetes
k8s-deploy:
	kubectl apply -f k8s/

# Clean up
clean:
	$(ENGINE) system prune -f
	$(ENGINE) volume prune -f

# Run tests
test:
	python3 -m pytest tests/

# Install dependencies
install:
	cd describe && python3 -m venv ./venv && \
	. ./venv/bin/activate && \
	pip install -r requirements.txt

# Example usage:
# make ENGINE=docker build
# make ENGINE=podman build
# make run
# make push-docker
# make push-podman
# make k8s-deploy

.PHONY: all pip deps install build deploy deploy-amd64 docker-build venv upgrade upgrade-describe;
