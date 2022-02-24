# Setup instructions for the GCP deep learning docker services.

This repository includes `make` commands for launching deep learning model
training on Google Cloud Platform (GCP). For these commands to work, you must
have a GCP account and set up your GCP environment variables in a `.env` file in
this directory.  A template is provided in `.env.template`.

## Input data

You must also have training data in `tfrec` format available in the Google Cloud
Storage (GCS) bucket specified in the `.env` file. You can download all of the
tfrec files using the `make tfrecs` command. You can further decompose these
tfrecs into multiple train, validate, test, and report subsets using the
`2021-08-24-cloudknot-save-tfrecs.ipynb` jupyter notebook in the "notebooks"
directory. You must then store these different splits in the GCS bucket using
the following scheme:

    .
    ├── tfrecs
    │   ├── b0-tensorfa-dwiqc
    │   │   ├── all-data
    │   │   │   └── data-all_shard-*.tfrec      <- Glob representing all data shards
    │   │   ├── seed_0
    │   │   │   ├── data-report_shard-*.tfrec   <- Glob representing the report set
    │   │   │   ├── data-test_shard-*.tfrec     <- Glob representing the test set
    │   │   │   ├── data-train_shard-*.tfrec    <- Glob representing the training set
    │   │   │   └── data-validate_shard-*.tfrec <- Glob representing the validation set
    │   │   ├── seed_1                          <- Data decomposition with different seed
    │   │   │   └── ...
    │   │   ├── seed_2                          <- Data decomposition with different seed
    │   │   │   └── ...
    │   │   └── ...
    │   ├── CBIC
    │   │   ├── data-CBIC-report_shard-*.tfrec
    │   │   ├── data-CBIC-test_shard-*.tfrec
    │   │   ├── report-subjects.csv
    │   │   ├── test-subjects.csv
    │   │   ├── seed_0
    │   │   │   ├── data-CBIC-train_shard-*.tfrec
    │   │   │   ├── data-CBIC-validate_shard-*.tfrec
    │   │   │   ├── train-subjects.csv
    │   │   │   ├── validate-subjects.csv
    │   │   │   └── num_volumes.csv
    │   │   ├── seed_1
    │   │   │   └── ...
    │   │   ├── seed_2
    │   │   │   └── ...
    │   │   └── ...
    │   ├── CBIC_CUNY
    │   │   └── ...
    │   ├── RU
    │   │   └── ...
    │   ├── RU_CUNY
    └   └   └── ...

The sub-directory "b0-tensorfa-dwiqc" is required for the `make dl-train`, `make
dl-predict`, and `make dl-integrated-gradients` commands, while the site
specific sub-directories "CBIC," "CBIC_CUNY," "RU," and "RU_CUNY" are required
for the `make dl-site-generalization` command. To generate the site specific
sub-directories, use the `2022-01-03-cloudknot-save-site-tfrecs.ipynb` jupyter
notebook in the "notebooks" directory.