.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE := default
PROJECT_NAME := albp_solver

# read variables from .env
ifneq ($(wildcard .env),)
	include .env
	export
endif

INCLUDE_DIR := $($(PYTHON_INTERPRETER) -c "import sysconfig; print(sysconfig.get_path('include'))")
LIBRARY := $($(PYTHON_INTERPRETER) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
ifeq ($(ENVIRONMENT), PRODUCTION)
	export PYTHONPATH=$(PROJECT_DIR)/myenv/lib/python3.9/site-packages:$(PYTHONPATH) && \
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt && \
	CMAKE_ARGS="-DSCIP_DIR=$(SCIP_DIR) -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON" $(PYTHON_INTERPRETER) -m pip install ecole
else ifeq ($(ENVIRONMENT), DEVELOPMENT)
endif

## Make Dataset
data:
ifeq ($(ENVIRONMENT), PRODUCTION)
	export PYTHONPATH=$(PROJECT_DIR)/myenv/lib/python3.9/site-packages:$(PYTHONPATH) && \
	$(PYTHON_INTERPRETER) albp/data/make_dataset.py
else ifeq ($(ENVIRONMENT), DEVELOPMENT)
	$(PYTHON_INTERPRETER) -m albp.data.make_dataset gasse
endif

features:
ifeq ($(ENVIRONMENT), PRODUCTION)
	export PYTHONPATH=$(PROJECT_DIR)/myenv/lib/python3.9/site-packages:$(PYTHONPATH) && \
	$(PYTHON_INTERPRETER) -m albp.features.build_features gasse -c albp/features/config.yml
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 albp

## Set up python interpreter environment
create_environment:
ifeq ($(ENVIRONMENT), PRODUCTION)
ifeq ($(shell [ ! -d "myenv" ]),)
	@echo ">>> Installing production environment if not already installed."
	/home/$(USER)/Python-3.9.18/python -m ensurepip --upgrade
	/home/$(USER)/Python-3.9.18/python -m venv myenv
	export PYTHONPATH=$(PWD)/myenv/lib/python3.9/site-packages:$(PYTHONPATH)
	source myenv/bin/activate && \
	python3 -m pip install --upgrade pip
	@echo ">>> New virtualenv created."
endif
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

train:
ifeq ($(ENVIRONMENT), PRODUCTION)
	export PYTHONPATH=$(PROJECT_DIR)/myenv/lib/python3.9/site-packages:$(PYTHONPATH) && \
	$(PYTHON_INTERPRETER) -m albp.models.train_model gasse -c models/gasse/config.yml
endif
	$(PYTHON_INTERPRETER) -m albp.models.train_model gasse -c models/gasse/config.yml


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
