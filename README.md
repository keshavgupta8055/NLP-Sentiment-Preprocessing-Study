# Impact of Preprocessing Choices on Classical NLP Sentiment Analysis

**Authors:** Mudassir Iftikhar & Keshav Gupta  
**Dataset:** IMDB Movie Reviews (50,000 reviews)  
**Model:** Logistic Regression + TF-IDF

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Experiments](#experiments)
- [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Key Findings](#key-findings)
- [Deliverables](#deliverables)

---

## Overview

This project investigates how different **text preprocessing choices** affect the performance of a classical NLP sentiment classifier on the IMDB movie reviews dataset.

Rather than treating preprocessing as a fixed step, we treat it as an **experimental variable**. Five pipelines are tested — each isolating the effect of one preprocessing component — using **TF-IDF** for feature extraction and **Logistic Regression** as the sole classifier.

We then run a **hyperparameter grid search** over vocabulary size (`max_features`) and regularisation strength (`C`) to find the optimal model configuration.

**Final best model:** `Exp5 pipeline` + `max_features=50,000` + `C=5.0`  
→ **Accuracy: 90.85%** | **F1-Score: 0.9090**

---

## Project Structure

```
├── Final_NLP_Project.ipynb     # Main notebook — all experiments + tuning
├── figs/                       # All output plots (
│   ├── accuracy_bar.png
│   ├── f1_bar.png
│   ├── confusion_matrix.png
│   ├── heatmap.png
│   ├── sensitivity.png
│   ├── sentiment_dist.png
│   └── review_length.png
├── mini-project-report.pdf             
├── NLP-PPT.pptx          
└── README.md               
```

---


### Download the dataset

The IMDB dataset is not included in this repo due to size. Download it from:

- [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

After downloading, place `IMDB.csv` in the **same folder** as the notebook.

### Download NLTK resources

The notebook handles this automatically on first run. If you prefer to download manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## How to Run

1. Clone or download this repository
2. Place `IMDB.csv` in the project folder
3. Open `Final_NLP_Project.ipynb` in Jupyter or VS Code
4. Run all cells top to bottom — **each step is clearly labelled**

The notebook is divided into 15 steps:

| Step | What it does |
|------|-------------|
| 1 | Import libraries and set global constants |
| 2 | Download NLTK resources |
| 3 | Load and normalise the dataset |
| 4 | EDA — missing values, duplicates, review length distribution |
| 5 | Class balance check |
| 6 | Define individual preprocessing functions |
| 7 | Define 5 experiment pipelines |
| 8 | Preview each preprocessing step on sample reviews |
| 9 | Train/test split (80/20 stratified) |
| 10 | Helper functions (TF-IDF vectorizer, metrics, plots) |
| 11 | Run all 5 experiments with Logistic Regression |
| 12 | Results summary table |
| 13 | Comparison bar charts (accuracy, F1, precision, recall) |
| 14 | Best model confusion matrix + detailed analysis |
| 15 | Hyperparameter tuning grid (max_features × C) |

---

## Experiments

All five pipelines share the same **base steps**:
1. Lowercase text
2. Remove punctuation
3. Tokenise (NLTK `punkt`)

Each experiment then adds a different combination on top:

| Experiment | Additional Steps | TF-IDF N-gram | Purpose |
|-----------|-----------------|--------------|---------|
| **Exp1** | None (baseline) | Unigrams (1,1) | Minimal preprocessing baseline |
| **Exp2** | + Stopword removal | Unigrams (1,1) | Isolate stopword effect |
| **Exp3** | + Porter stemming | Unigrams (1,1) | Isolate stemming effect |
| **Exp4** | + Stopwords & stemming | Unigrams (1,1) | Combined effect |
| **Exp5** | + Stopwords & bigrams | Unigrams+Bigrams (1,2) | N-gram range effect |

**Model:** Logistic Regression (`C=1.0`, solver=`lbfgs`, `max_iter=1000`)  
**Features:** TF-IDF with `max_features=10,000`, `sublinear_tf=True`

---

## Results

All five experiments exceeded **89% accuracy**, confirming TF-IDF + Logistic Regression is a strong baseline.

| Experiment | Accuracy | Precision | Recall | F1-Score |
|-----------|---------|-----------|--------|---------|
| **Exp5: SW + Bigrams** | **0.8985** | 0.8906 | 0.9086 | **0.8995** |
| Exp1: Baseline | 0.8968 | 0.8913 | 0.9038 | 0.8975 |
| Exp2: +Stopwords | 0.8961 | 0.8884 | 0.9060 | 0.8971 |
| Exp3: +Stemming | 0.8947 | 0.8879 | 0.9034 | 0.8956 |
| Exp4: SW+Stemming | 0.8934 | 0.8851 | 0.9042 | 0.8945 |

### Best Model Confusion Matrix (Exp5)

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 4,442 (TN) | 558 (FP) |
| **Actual Positive** | 457 (FN) | 4,543 (TP) |

- **Specificity:** 0.8884
- **Negative Predictive Value (NPV):** 0.9067

---

## Hyperparameter Tuning

We ran a **grid search** on the best pipeline (Exp5) over:

- `max_features` ∈ {5,000 · 10,000 · 20,000 · 50,000}
- `C` ∈ {0.1 · 0.5 · 1.0 · 5.0 · 10.0}

**20 combinations total.**

### Top 5 Results

| max_features | C | Accuracy | F1-Score |
|-------------|---|---------|---------|
| **50,000** | **5.0** | **0.9085** | **0.9090** |
| 50,000 | 10.0 | 0.9070 | 0.9074 |
| 20,000 | 5.0 | 0.9056 | 0.9063 |
| 20,000 | 1.0 | 0.9037 | 0.9031 |
| 20,000 | 10.0 | 0.9030 | 0.9029 |

**Improvement over default Exp5:**  
→ `+0.010` accuracy | `+0.0095` F1-score

---

## Key Findings

1. **A basic pipeline is very competitive** — Exp1 (minimal cleaning) ranked 2nd overall. More preprocessing is not always better.

2. **Stemming consistently hurts** — Porter stemmer is too aggressive for sentiment tasks. It merges words like *good* and *goods* that carry different meanings, destroying useful signal.

3. **Stopword removal is nearly neutral** — Words like *not* and *never* are classified as stopwords but carry strong sentiment. Removing them cancels out any noise reduction benefit.

4. **Bigrams gave the biggest preprocessing gain** — Capturing two-word phrases like *"not good"* or *"highly recommended"* that unigrams miss entirely is the most impactful single change.

5. **Vocabulary size is the most important tuning parameter** — Increasing `max_features` from 10k to 50k improved average F1 by ~+0.006. Tuning `C` contributed only ~+0.002.

---

## Deliverables

| File | Description |
|------|-------------|
| `Final_NLP_Project.ipynb` | Full annotated notebook with all experiments and tuning |
| `mini-project-report.pdf` | 3-page ACL two-column format research report |
| `NLP-PPT.pptx` | 5-slide PowerPoint presentation |
| `figs/*.png` | All 7 high-resolution plots |





---

*Project submitted as part of an NLP mini project — Mudassir Iftikhar & Keshav Gupta*
