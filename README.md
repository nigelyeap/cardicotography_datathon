# CTG Fetal Health Classification

Team TM-207: Alyssa, Nigel, Louis

Before we begin, we would like to emphasise that most of our accurate code can be found under data_exploration/mainanalysis, as we were un
## Overview

This project predicts fetal health status (Normal, Suspect, or Pathological) from CTG data using a Random Forest model with SMOTE-Tomek resampling. Achieved 92% test accuracy.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run predictions on your data

```bash
python test.py --input your_data.xlsx --output results.csv
```

That's it! Your predictions will be saved in `results.csv` with two columns: `Sample_ID` and `Predicted_Class` (1=Normal, 2=Suspect, 3=Pathological).

## Example Usage

```bash
# Basic prediction
python test.py --input test_data.xlsx --output predictions.csv

# Evaluate accuracy (if your file has an NSP column with ground truth)
python test.py --input test_data.xlsx --output predictions.csv --evaluate
```

## Repository Structure

```
├── data_exploration/        # EDA notebooks and raw data
├── train.py                 # Model training script
├── test.py                  # Prediction script
├── transformedCTG.xlsx      # Processed training data
├── trained_model.pkl        # Trained model
├── scaler.pkl              # Feature scaler
├── model_metadata.json     # Training details
├── model_weights.npz       # Model weights
└── requirements.txt        # Python dependencies
```

## Training Your Own Model

If you want to retrain the model:

```bash
python train.py
```

This will generate new model files (`trained_model.pkl`, `scaler.pkl`, etc.).

## EDA

Our exploratory data analysis and model development process can be found in the `data_exploration/` folder.

## Model Details

- Algorithm: Random Forest (600 trees)
- Resampling: SMOTE-Tomek for class imbalance
- Features: 11 CTG parameters
- Classes: Normal (1), Suspect (2), Pathological (3)
- Test Accuracy: 92%

## Notes

- The model was trained on 2,126 CTG samples
- SMOTE-Tomek handles the class imbalance (77.8% normal, 13.9% suspect, 8.3% pathological)
- Predictions include confidence scores displayed in the console output
