default: help
.PHONY: help build data niftis tfrecs
.PHONY: expert-qc community-qc
.PHONY: deep-learning-figures bundle-profiles inference
.PHONY: dl-train dl-predict dl-site-generalization dl-integrated-gradients

# Show this help message
help:
	@cat $(MAKEFILE_LIST) | docker run --rm -i xanders/make-help

all: build data expert-qc community-qc deep-learning-figures bundle-profiles

# Build all of the necessary docker images
build:
	@echo "Building all of the necessary docker images"
	@docker inspect hbn-pod2/base:conda-tex > /dev/null 2>&1 && echo "Base image already exists" || docker buildx build --platform=linux/amd64 -t hbn-pod2/base:conda-tex -f docker/base/Dockerfile docker/base
	@docker compose build

# Download data from OSF
data:
	@echo "Downloading data from OSF"
	@docker compose run osf-download
	@echo "This download excluded the NIfTI files and TFRecords, which are large and can be time consuming to download."
	@echo "To download those files, use the make niftis and make tfrecs commands, respectively."

# Download nifti files from OSF
niftis:
	@echo "Downloading NIfTI files from FCP-INDI"
	@docker compose run nifti-download

# Download tfrecs files from OSF
tfrecs:
	@echo "Downloading tfrecs from FCP-INDI"
	@docker compose run tfrec-download

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

# Plot bundle profiles binned by QC score
bundle-profiles:
	@echo "Plotting bundle profiles binned by QC score"
	@docker compose run bundle-profiles

# Demonstrate the effect of QC on inference using an age prediction example
inference:
	@echo "Demonstrating the effect of QC on inference using an age prediction example"
	@docker compose run inference
	
# Train site generalization models and compute performance metrics
site-generalization:
	@echo "Training site generalization models and computing performance metrics"
	@docker compose run site-generalization

##
## Commands for launching deep learning model training on GCP. For these commands to work, you must have a GCP account and set up your GCP environment variables in a .env file in this directory.  A template is provided in .env.template. For further details see README_GCP.md.
##

# Train the deep learning model on GCP
dl-train:
	@echo "Launching deep learning model training on GCP"
	@docker compose run dl-train-gcp b0_tensorfa_dwiqc

# Predict QC ratings using the trained models on GCP
dl-predict:
	@echo "Launching deep learning model prediction on GCP"
	@docker compose run dl-predict-gcp b0_tensorfa_dwiqc

# Train site generalization models on GCP
dl-site-generalization:
	@echo "Launching site generalization model training on GCP"
	@docker compose run dl-site-generalization-gcp

# Generate attribution maps using integrated gradients on GCP
dl-integrated-gradients:
	@echo "Launching integrated gradients on GCP"
	@docker compose run dl-integrated-gradients-gcp site_gen
