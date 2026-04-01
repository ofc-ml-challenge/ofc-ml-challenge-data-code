# OFC ML Challenge — Results, Baseline Code, and Winning Solution

This repository contains:

- the ML Challenge results and leaderboard
- the baseline feature-extraction and training pipeline
- the winning solution summary and links

The baseline workflow consists of two notebooks:

1. `code/kaggle_feature_extraction_admin.ipynb` builds train/test/TestGround CSV files from raw JSON measurement data.
2. `code/ML_example_kaggle.ipynb` trains a DNN baseline model and generates predictions for the test set.

The repository also includes `dataset/COSMOS_EDFA_Dataset`, which provides additional EDFA measurement data that can optionally be used for extended training and evaluation.

---

## Challenge Results

### Status

The ML Challenge has concluded. Submissions are closed.

### Final Results

- **Final announcement date:** 03/16/2026
- **Finalists:**
  - SJTU (Shanghai Jiao Tong University)
  - IPOC (Beijing University of Posts and Telecommunications)
  - SIL (University of Bristol)
  - NU (Nagoya University and Toyota Technological Institute)
- **Winner:** SJTU (Shanghai Jiao Tong University)
- **Event page:** [OFC ML Challenge Event Page](https://ofc-ml-challenge.github.io/ofc-ml-challenge/)

### Live Competition Scores

The final live competition results are reported for `aging`, `shb` (spectral hole burning), `unseen`, and the overall score.

The score is computed on loaded channels only. It combines the mean absolute error (MAE), the standard deviation of absolute errors, and soft penalties for heavy-tail and extreme outlier errors based on the 95th-percentile and maximum error. Lower scores indicate better performance.

| Team | Aging    | SHB      | Unseen   | Overall  | Final Rank |
|:----:|:--------:|:--------:|:--------:|:--------:|:----------:|
| SJTU | 0.130171 | 0.079764 | 0.098675 | 0.108047 | 1          |
| IPOC | 0.121494 | 0.112967 | 0.115511 | 0.117331 | 2          |
| SIL  | 0.121167 | 0.070625 | 0.124217 | 0.121183 | 3          |
| NU   | 0.116429 | 0.072435 | 0.313027 | 0.240763 | 4          |

### Winning Solution

- **Winning team:** SJTU (Shanghai Jiao Tong University)
- **Summary:** SJTU proposed a global spectral mapping strategy for EDFA gain-profile prediction based on FourierKAN operator learning. Their method combines pretraining on converted COSMOS data with fine-tuning on the challenge dataset, models the gain profile as a linear gain/tilt baseline plus a learned residual ripple, and applies a WSS-based loss mask so that only active channels contribute during training.
- **Team code repository:** [SJTU GitHub Repository](https://github.com/yukangZhuu/ofc_ml.git)
- **Winning report (PDF):** [SJTU Winning Report PDF](https://drive.google.com/file/d/1Oqnwf9rqTXCEpvPPRNrCn5JlXeS5kmw5/view?usp=sharing)

---

## Environment

- **Recommended Python version:** 3.11

Install the environment with:

```bash
pip install -U pip
pip install -r requirements.txt
```

If you prefer a minimal manual install, use:

```bash
pip install pandas numpy matplotlib tensorflow prettytable scipy scikit-learn
```

The feature extraction notebook imports `libs.edfa_feature_extraction_libs` and `libs.edfaBasicLib`.

---

## Repository Layout

```text
.
├── requirements.txt
├── README.md
├── LICENSE
├── code
│   ├── kaggle_feature_extraction_admin.ipynb
│   ├── ML_example_kaggle.ipynb
│   └── libs
│       ├── edfaBasicLib.py
│       └── edfa_feature_extraction_libs.py
├── dataset
│   ├── COSMOS_EDFA_Dataset
│   │   ├── booster/{15dB,18dB,21dB}/...
│   │   └── preamp/{15dB,18dB,21dB,24dB,27dB}/...
│   └── ML_challenge_admin
│       ├── Train/{aging,shb,unseen}/*.json
│       ├── Test/{aging,shb,unseen}/*.json
│       └── TestGround/{aging,shb,unseen}/*.json
├── Features
│   ├── Test
│   │   ├── test_features.csv
│   │   ├── aging/features/*.csv
│   │   ├── shb/features/*.csv
│   │   └── unseen/features/*.csv
│   ├── TestGround
│   │   ├── test_features.csv
│   │   ├── test_labels.csv
│   │   ├── example_submission.csv
│   │   ├── aging/{features,labels}/*.csv
│   │   ├── shb/{features,labels}/*.csv
│   │   └── unseen/{features,labels}/*.csv
│   └── Train
│       ├── train_features.csv
│       ├── train_labels.csv
│       ├── COSMOS_features.csv
│       ├── COSMOS_labels.csv
│       ├── aging/{features,labels}/*.csv
│       ├── shb/{features,labels}/*.csv
│       └── unseen/{features,labels}/*.csv
├── figures
├── manual
│   └── Ranges_ProductGuide_ROADM.pdf
└── model
    └── ML_example_model.h5
```

Notes:

- `dataset/ML_challenge_admin/Test` is the competition-style test set without exposed labels.
- `dataset/ML_challenge_admin/TestGround` is the corresponding ground-truth version of the test set, with labels available.
- `Features/Test/test_features.csv` contains the combined features for the label-hidden test set.
- `Features/TestGround/test_features.csv` and `Features/TestGround/test_labels.csv` contain the combined feature and label tables for the ground-truth test set.
- `Features/TestGround/example_submission.csv` is a Kaggle-style label file derived from `Features/TestGround/test_labels.csv`.
- `dataset/COSMOS_EDFA_Dataset` and the corresponding `COSMOS_*.csv` files are optional.

---

## Baseline Pipeline

### `code/kaggle_feature_extraction_admin.ipynb`

- **Input data:** raw JSON files from `dataset/ML_challenge_admin/Train/{shb,aging,unseen}`, `dataset/ML_challenge_admin/Test/{shb,aging,unseen}`, and `dataset/ML_challenge_admin/TestGround/{shb,aging,unseen}`
- **Optional input data:** `dataset/COSMOS_EDFA_Dataset/{booster,preamp}/*/*.json`
- **Feature extraction:** uses `featureExtraction_ML` and helper utilities in `code/libs`
- **Metadata inference:** infers EDFA type, channel type, and EDFA name/index mapping
- **Outputs:**
  - per-dataset intermediate CSV files
  - `Features/Train/train_features.csv`
  - `Features/Train/train_labels.csv`
  - `Features/Test/test_features.csv`
  - `Features/TestGround/test_features.csv`
  - `Features/TestGround/test_labels.csv`
  - `Features/TestGround/example_submission.csv`
  - `Features/Train/COSMOS_features.csv` and `Features/Train/COSMOS_labels.csv` if COSMOS data is included

### `code/ML_example_kaggle.ipynb`

- **Inputs:** `Features/Train/train_features.csv`, `Features/Train/train_labels.csv`, and `Features/Test/test_features.csv`
- **Optional inputs:** `Features/Train/COSMOS_features.csv` and `Features/Train/COSMOS_labels.csv`
- **Model:** fully connected DNN baseline in TensorFlow / Keras
- **Loss:** custom L2 loss computed only on loaded channels
- **Training setup:** Adam optimizer, 500 epochs, 20% validation split
- **Outputs:**
  - `model/ML_example_model.h5`
  - `figures/*.png` if plotting cells are run
  - a prediction file for the label-hidden test set, in the same schema as the ground-truth submission format

---

## Run the Baseline

1. Open the repository in VS Code.
2. Open `code/kaggle_feature_extraction_admin.ipynb`.
3. Select the correct Python kernel.
4. Run all cells to generate feature and label CSV files under `Features/Train`, `Features/Test`, and `Features/TestGround`.
5. If you want to include COSMOS data, adjust the notebook accordingly before generating the combined files.
6. Open `code/ML_example_kaggle.ipynb`.
7. Run all cells to train the baseline model and generate predictions for `Features/Test/test_features.csv`.

Expected outputs:

- `Features/Train/train_features.csv`
- `Features/Train/train_labels.csv`
- `Features/Test/test_features.csv`
- `Features/TestGround/test_features.csv`
- `Features/TestGround/test_labels.csv`
- `Features/TestGround/example_submission.csv`
- `model/ML_example_model.h5`
- `figures/*.png` if plotting cells are executed
- `Features/Train/COSMOS_features.csv` and `Features/Train/COSMOS_labels.csv` if COSMOS data is used
