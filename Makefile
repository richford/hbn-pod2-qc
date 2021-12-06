default: help

# Show this help message
help:
	@cat $(MAKEFILE_LIST) | docker run --rm -i xanders/make-help

all: build data expert-qc community-qc deep-learning-figures

# Build all of the necessary docker images
build:
	@echo "Building all of the necessary docker images"
	@docker compose build

# Download data from OSF
data:
	@echo "Downloading data from OSF"
	@docker compose run osf-download

# Analyze expert ratings and generate derived figures
expert-qc:
	@echo "Analyzing expert ratings and generating derived figures"
	@docker compose run expert-qc

# Analyze community ratings and generate derived figures
community-qc:
	@echo "Analyzing community ratings and generating derived figures"
	@docker compose run community-qc

# Plot figures for the deep learning QC pipeline
deep-learning-figures:
	@echo "Plotting figures for the deep learning QC pipeline"
	@docker compose run dl-figures 

##
## Commands for launching deep learning model training on GCP. For these commands to work, you must have a GCP account and set up your GCP environment variables in a .env file in this directory.  A template is provided in .env.template.
##

# Train the deep learning model on GCP
dl-train:
	@echo "Launching deep learning model training on GCP"
	@docker compose run dl-train-gcp

# Predict QC ratings using the trained models on GCP
dl-predict:
	@echo "Launching deep learning model prediction on GCP"
	@docker compose run dl-predict-gcp

# Generate attribution maps using integrated gradients on GCP
dl-integrated-gradients:
	@echo "Launching integrated gradients on GCP"
	@docker compose run dl-integrated-gradients-gcp