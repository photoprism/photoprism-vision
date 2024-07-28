export VIRTUAL_ENV := $(abspath "./venv")
export VIRTUAL_ENV_BIN := $(abspath "./venv/bin")
export PATH := $(VIRTUAL_ENV_BIN):$(PATH)

BUILD_GIT ?= $(shell (cd .. && git describe --always))
BUILD_DATE ?= $(shell date -u +%y%m%d)
BUILD_TAG ?= $(BUILD_DATE)-$(BUILD_GIT)

UNAME := $(shell uname)

all: deps
deps: pip upgrade
install: install-venv
upgrade: remove-venv install-venv

build: docker-build
docker-build:
	docker compose --profile=all pull --ignore-pull-failures
	docker compose build

terminal:
	docker compose exec vision-describe bash
logs:
	docker compose logs -f

pip:
ifeq ($(UNAME), Linux)
	sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
endif

venv: install-venv
remove-venv:
	rm -rf ./venv
install-venv:
	python3 -m venv venv
	. ./venv/bin/activate
	./venv/bin/pip3 install --disable-pip-version-check -r requirements.txt

.PHONY: all pip deps upgrade build docker-build remove-venv build deploy;