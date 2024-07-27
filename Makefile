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

.PHONY: all pip deps install build docker-build venv upgrade upgrade-describe;
