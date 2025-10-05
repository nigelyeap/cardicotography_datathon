"""
CTG Fetal Health Classification - Testing Script
Team TM-207: Alyssa, Nigel, Louis
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pathlib import Path
from datetime import datetime

MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
METADATA_PATH = "model_metadata.json"
TEST_DATA_PATH = "transformedCTG.xlsx"

CLASS_LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}


def load_model_artifacts():
    print("=" * 60)
    print("Loading model artifacts...")
    print("=" * 60)
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from: {MODEL_PATH}")
        
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Scaler loaded from: {SCALER_PATH}")
        
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded from: {METADATA_PATH}")
        
        print("\nModel Information:")
        print("-" * 60)
        print(f"  Model Type: {metadata['model_type']}")
        print(f"  Preprocessing: {metadata['preprocessing']}")
        print(f"  Training Date: {metadata['training_date']}")
        print(f"  Test Accuracy: {metadata['performance']['test']['accuracy']:.4f}")
        print(f"  Features: {len(metadata['features'])}")
        
        return model, scaler, metadata
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: Required file not found - {e}")
        print("\nPlease run train.py first to generate model artifacts.")
        exit(1)


def load_test_data(file_path):
    print("\n" + "=" * 60)
    print("Loading test data...")
    print("=" * 60)
    
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
        
        print(f"✓ Loaded {len(df)} samples from: {file_path}")
        return df
    
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        exit(1)


def preprocess_input(df, scaler, feature_columns):
    print("\n" + "=" * 60)
    print("Preprocessing input data...")
    print("=" * 60)
    
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        print(f"\n✗ Error: Missing required features: {missing_features}")
        print(f"\nRequired features: {feature_columns}")
        exit(1)
    
    X = df[feature_columns].astype('float64')
    
    if X.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Filling with median values.")
        X = X.fillna(X.median())
    
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_columns,
        index=X.index
    )
    
    print(f"✓ Preprocessed {len(X_scaled)} samples")
    print(f"✓ Feature shape: {X_scaled.shape}")
    
    return X_scaled


def make_predictions(model, X):
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    print(f"✓ Generated predictions for {len(X)} samples")
    
    return y_pred, y_proba


def format_predictions(y_pred, y_proba, input_df=None):
    results = pd.DataFrame()
    
    if input_df is not None and len(input_df) == len(y_pred):
        if 'sample_id' in input_df.columns:
            results['Sample_ID'] = input_df['sample_id']
        else:
            results['Sample_ID'] = range(1, len(y_pred) + 1)
    else:
        results['Sample_ID'] = range(1, len(y_pred) + 1)
    
    results['Predicted_Class'] = y_pred.astype(int)
    
    results['_Predicted_Label'] = results['Predicted_Class'].map(CLASS_LABELS)
    results['_Prob_Normal'] = (y_proba[:, 0] * 100).round(2)
    results['_Prob_Suspect'] = (y_proba[:, 1] * 100).round(2)
    results['_Prob_Pathological'] = (y_proba[:, 2] * 100).round(2)
    results['_Confidence'] = results[['_Prob_Normal', '_Prob_Suspect', '_Prob_Pathological']].max(axis=1)
    
    def assess_risk(row):
        if row['Predicted_Class'] == 1 and row['_Confidence'] > 90:
            return "Low Risk"
        elif row['Predicted_Class'] == 1:
            return "Low-Moderate Risk"
        elif row['Predicted_Class'] == 2:
            return "Moderate-High Risk"
        else:
            return "High Risk"
    
    results['_Risk_Assessment'] = results.apply(assess_risk, axis=1)
    
    return results


def display_summary(results):
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    print("\nPredicted Class Distribution:")
    print("-" * 60)
    class_counts = results['_Predicted_Label'].value_counts()
    for label, count in class_counts.items():
        pct = (count / len(results)) * 100
        print(f"  {label}: {count} samples ({pct:.1f}%)")
    
    print("\nRisk Assessment Distribution:")
    print("-" * 60)
    risk_counts = results['_Risk_Assessment'].value_counts()
    for risk, count in risk_counts.items():
        pct = (count / len(results)) * 100
        print(f"  {risk}: {count} samples ({pct:.1f}%)")
    
    print("\nConfidence Statistics:")
    print("-" * 60)
    print(f"  Mean Confidence: {results['_Confidence'].mean():.2f}%")
    print(f"  Min Confidence: {results['_Confidence'].min():.2f}%")
    print(f"  Max Confidence: {results['_Confidence'].max():.2f}%")
    
    high_risk = results[results['Predicted_Class'] == 3]
    if len(high_risk) > 0:
        print(f"\n⚠️  Warning: {len(high_risk)} PATHOLOGICAL cases detected!")
        print("    Immediate medical attention recommended.")
    
    suspect = results[results['Predicted_Class'] == 2]
    if len(suspect) > 0:
        print(f"\n⚠️  Caution: {len(suspect)} SUSPECT cases detected!")
        print("    Further monitoring recommended.")


def evaluate_predictions(results, true_labels):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report, confusion_matrix
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    y_true = true_labels.astype(int)
    y_pred = results['Predicted_Class'].values
    
    print("\nOverall Performance:")
    print("-" * 60)
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Suspect', 'Pathological']))


def save_predictions(results, output_path):
    print("\n" + "=" * 60)
    print("Saving predictions...")
    print("=" * 60)
    
    try:
        output_df = results[['Sample_ID', 'Predicted_Class']].copy()
        
        if output_path.endswith('.xlsx'):
            output_df.to_excel(output_path, index=False)
        elif output_path.endswith('.csv'):
            output_df.to_csv(output_path, index=False)
        else:
            output_path = output_path + '.csv'
            output_df.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Error saving predictions: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='CTG Fetal Health Classification - Testing Script'
    )
    parser.add_argument('--input', '-i', type=str, 
                       help='Path to input file (Excel or CSV)')
    parser.add_argument('--output', '-o', type=str, 
                       help='Path to output file for predictions')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions against ground truth (requires NSP column)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CTG FETAL HEALTH CLASSIFICATION - TESTING")
    print("Team TM-207: Alyssa, Nigel, Louis")
    print("=" * 60)
    
    model, scaler, metadata = load_model_artifacts()
    feature_columns = metadata['features']
    
    input_file = args.input if args.input else TEST_DATA_PATH
    
    df = load_test_data(input_file)
    
    has_ground_truth = 'NSP' in df.columns
    if has_ground_truth:
        true_labels = df['NSP'].copy()
    
    X_scaled = preprocess_input(df, scaler, feature_columns)
    
    y_pred, y_proba = make_predictions(model, X_scaled)
    
    results = format_predictions(y_pred, y_proba, df)
    
    display_summary(results)
    
    if has_ground_truth and args.evaluate:
        evaluate_predictions(results, true_labels)
    
    if args.output:
        save_predictions(results, args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"predictions_{timestamp}.csv"
        save_predictions(results, output_file)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Review predictions in the output file")
    print("  2. For high-risk cases, recommend immediate medical consultation")
    print("  3. For suspect cases, recommend continued monitoring")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
