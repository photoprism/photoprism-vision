BUILD_GIT ?= $(shell (cd .. && git describe --always))
BUILD_DATE ?= $(shell date -u +%y%m%d)
BUILD_TAG ?= $(BUILD_DATE)-$(BUILD_GIT)

UNAME := $(shell uname)

all: pip install
deps: pip upgrade
install: venv

build: docker-build
docker-build:
	(cd service && make docker-build)

preview: docker-preview
docker-preview:
	./build.sh service linux/amd64

release: docker-release
docker-release:
	./build.sh service linux/amd64,linux/arm64 $(BUILD_DATE)

ollama:
	docker compose pull ollama
	docker compose up -d ollama --remove-orphans
	docker compose exec ollama ollama run llama3.2-vision
ollama-down:
	docker compose down ollama --remove-orphans
start:
	docker compose --profile=all pull --ignore-pull-failures
	docker compose up -d
	docker compose logs -f || true
stop:
	docker compose down -v
terminal:
	docker compose exec photoprism-vision bash
logs:
	docker compose logs -f || true

pip:
ifeq ($(UNAME), Linux)
	sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
endif

venv: service
service/venv:
	make -C service venv

upgrade:
	make -C service upgrade

# Declare all targets as "PHONY", see https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html.
MAKEFLAGS += --always-make
