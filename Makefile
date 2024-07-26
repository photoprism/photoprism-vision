BUILD_GIT ?= $(shell (cd .. && git describe --always))
BUILD_DATE ?= $(shell date -u +%y%m%d)
BUILD_TAG ?= $(BUILD_DATE)-$(BUILD_GIT)

all: describe-amd64
describe-amd64:
	./build.sh describe linux/amd64
describe:
	./build.sh describe linux/amd64,linux/arm64

# Declare all targets as "PHONY", see https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html.
MAKEFLAGS += --always-make