BASE_IMAGE = ghcr.io/allenai/pytorch:2.4.0-cuda12.1-python3.11
# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
NIGHTLY_VERSION = "2.5.0.dev20240826+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121"

VERSION = $(shell python src/flexolmo/version.py)
VERSION_SHORT = $(shell python src/flexolmo/version.py short)
IMAGE_BASENAME = flexolmo
BEAKER_WORKSPACE = ai2/flexolmo
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : checks
checks : style-check lint-check type-check

.PHONY : style-check
style-check :
	@echo "======== running isort... ========"
	@isort --check .
	@echo "======== running black... ========"
	@black --check .

.PHONY : lint-check
lint-check :
	@echo "======== running ruff... ========="
	@ruff check .

.PHONY : type-check
type-check :
	@echo "======== running mypy... ========="
	@mypy src/

.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/flexolmo/ --watch README.md docs/source/ docs/build/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

.PHONY : stable-image
stable-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg BASE=$(BASE_IMAGE) \
		--target stable \
		-t $(IMAGE_BASENAME) .

.PHONY : beaker-image-stable
beaker-image-stable : stable-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : nightly-image
nightly-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg BASE=$(BASE_IMAGE) \
		--build-arg NIGHTLY_VERSION=$(NIGHTLY_VERSION) \
		--target nightly \
		-t $(IMAGE_BASENAME)-nightly .

.PHONY : beaker-image-nightly
beaker-image-nightly : nightly-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION_SHORT)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION)-nightly $(BEAKER_WORKSPACE)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
