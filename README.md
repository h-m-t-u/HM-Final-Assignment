# The Refining of my Models for the Final Project Kaggle Competition

This repository is all mine. 

## Purpose

The layout of this project is the recommended directory organization shown in the assignment instructions. It is a reference. 

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   └── raw/
│       └── card_transdata.csv
├── notebooks/
│   └── credit_card_fraud_analysis.ipynb
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── split_data.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── train_model.py
    │   ├── dumb_model.py
    │   └── knn_model.py
    ├── utils/
    │   └── helper_functions.py
    └── visualization/
        ├── eda.py
        └── performance.py
```

