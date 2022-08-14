
THIS_FILE := $(lastword $(MAKEFILE_LIST))
FOLDER_PROJECT = ./
FOLDER_CONFIGS = scripts/configs/


.PHONY: clean
clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type f -name '*.coverage.*' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@find . -type d -name 'build' | xargs rm -rf
	@find . -type d -name 'dist' | xargs rm -rf
	@find . -type d -name '*.egg*' | xargs rm -rf
	@find . -type d -name 'docs/build/' | xargs rm -rf
	@find . -type d -name '.pytest_cache' | xargs rm -rf


.PHONY: build
build:
	@poetry shell
	@poetry install

update:
	@poetry update

black:
	@black $(FOLDER_PROJECT) --config $(FOLDER_CONFIGS)/pyproject.toml $(args)

lint:
	@find $(FOLDER_PROJECT) -type f -name "*.py" | xargs pylint --rcfile=${FOLDER_CONFIGS}/.pylintrc $(args)

isort:
	@isort $(FOLDER_PROJECT)

code-review:
	$(MAKE) -f $(THIS_FILE) isort args=--check-only
	$(MAKE) -f $(THIS_FILE) black args=--check
	$(MAKE) -f $(THIS_FILE) lint
