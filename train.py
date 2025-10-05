"""
CTG Fetal Health Classification - Training Script
Team TM-207: Alyssa, Nigel, Louis
i w"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.combine import SMOTETomek

RANDOM_STATE = 42
DATA_PATH = "transformedCTG.xlsx"
MODEL_OUTPUT_PATH = "trained_model.pkl"
SCALER_OUTPUT_PATH = "scaler.pkl"
METADATA_OUTPUT_PATH = "model_metadata.json"
WEIGHTS_OUTPUT_PATH = "model_weights.npz"

FEATURE_COLUMNS = ['LB', 'AC', 'MSTV', 'ASTV', 'DS', 'DP', 'DR', 'DL', 'Variance', 'Tendency']
TARGET_COLUMN = 'NSP'

MODEL_HYPERPARAMS = {
    'n_estimators': 600,
    'criterion': 'gini',
    'max_features': 'log2',
    'max_depth': 30,
    'min_samples_split': 20,
    'random_state': RANDOM_STATE
}


def load_and_preprocess_data(data_path):
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    df = pd.read_excel(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Dropping rows with missing values.")
        df = df.dropna()
    
    class_dist = df[TARGET_COLUMN].value_counts().sort_index()
    print("\nClass Distribution:")
    for cls, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"  Class {int(cls)}: {count} samples ({pct:.2f}%)")
    
    return df


def stratified_split(X, y, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    print("\n" + "=" * 60)
    print("Splitting data...")
    print("=" * 60)
    
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
    
    val_fraction = val_size / (train_size + val_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    print(f"Train set: {X_train.shape[0]} samples ({train_size*100:.0f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({val_size*100:.0f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote_tomek(X_train, y_train, random_state=42):
    print("\n" + "=" * 60)
    print("Applying SMOTE-Tomek resampling...")
    print("=" * 60)
    
    print(f"Original training set: {X_train.shape[0]} samples")
    
    smt = SMOTETomek(random_state=random_state)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
    
    print(f"Resampled training set: {X_train_resampled.shape[0]} samples")
    
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print("\nResampled Class Distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {int(cls)}: {count} samples")
    
    return X_train_resampled, y_train_resampled


def train_model(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("Training Random Forest Classifier...")
    print("=" * 60)
    
    print(f"Hyperparameters: {MODEL_HYPERPARAMS}")
    
    clf = RandomForestClassifier(**MODEL_HYPERPARAMS)
    clf.fit(X_train, y_train)
    
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return clf


def evaluate_model(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    sets = [
        ("Train", y_train, y_train_pred),
        ("Validation", y_val, y_val_pred),
        ("Test", y_test, y_test_pred)
    ]
    
    results = {}
    for set_name, y_true, y_pred in sets:
        print(f"\n{set_name} Set Metrics:")
        print("-" * 40)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        results[set_name.lower()] = metrics
    
    print("\n" + "=" * 60)
    print("Test Set Confusion Matrix:")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print("\nTest Set Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Normal', 'Suspect', 'Pathological']))
    
    print("\n" + "=" * 60)
    print("Top 5 Feature Importances:")
    print("=" * 60)
    feature_imp = pd.Series(clf.feature_importances_, 
                           index=FEATURE_COLUMNS).sort_values(ascending=False)
    for feature, importance in feature_imp.head().items():
        print(f"  {feature}: {importance:.4f}")
    
    return results


def extract_model_weights(clf):
    print("\n" + "=" * 60)
    print("Extracting model weights...")
    print("=" * 60)
    
    weights = {}
    weights['feature_importances'] = clf.feature_importances_
    weights['n_estimators'] = clf.n_estimators
    weights['n_classes'] = clf.n_classes_
    weights['n_features'] = clf.n_features_in_
    
    tree_predictions = []
    for i, tree in enumerate(clf.estimators_):
        tree_info = {
            'feature': tree.tree_.feature,
            'threshold': tree.tree_.threshold,
            'value': tree.tree_.value,
            'n_node_samples': tree.tree_.n_node_samples
        }
        tree_predictions.append(tree_info)
    
    print(f"✓ Extracted weights from {len(clf.estimators_)} trees")
    print(f"✓ Feature importances shape: {clf.feature_importances_.shape}")
    
    return weights, tree_predictions


def save_model_artifacts(clf, scaler, metadata, 
                        model_path, scaler_path, metadata_path, weights_path):
    print("\n" + "=" * 60)
    print("Saving model artifacts...")
    print("=" * 60)
    
    joblib.dump(clf, model_path)
    print(f"Model saved to: {model_path}")
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")
    
    weights, tree_predictions = extract_model_weights(clf)
    np.savez_compressed(weights_path, 
                       feature_importances=weights['feature_importances'],
                       n_estimators=weights['n_estimators'],
                       n_classes=weights['n_classes'],
                       n_features=weights['n_features'])
    print(f"Model weights saved to: {weights_path}")


def main():
    print("\n" + "=" * 60)
    print("CTG FETAL HEALTH CLASSIFICATION - TRAINING")
    print("Team TM-207: Alyssa, Nigel, Louis")
    print("=" * 60)
    
    training_start_time = datetime.now()
    
    df = load_and_preprocess_data(DATA_PATH)
    
    X = df[FEATURE_COLUMNS].astype('float64')
    y = df[TARGET_COLUMN].astype('float64')
    
    print("\n" + "=" * 60)
    print("Scaling features with RobustScaler...")
    print("=" * 60)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print("Feature scaling complete.")
    
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X_scaled, y, 
        train_size=0.7, 
        val_size=0.2, 
        test_size=0.1, 
        random_state=RANDOM_STATE
    )
    
    y_train = np.ravel(y_train)
    y_val = np.ravel(y_val)
    y_test = np.ravel(y_test)
    
    X_train_resampled, y_train_resampled = apply_smote_tomek(
        X_train, y_train, 
        random_state=RANDOM_STATE
    )
    
    clf = train_model(X_train_resampled, y_train_resampled, X_val, y_val)
    
    results = evaluate_model(clf, X_train, y_train, X_val, y_val, X_test, y_test)
    
    training_end_time = datetime.now()
    training_duration = (training_end_time - training_start_time).total_seconds()
    
    metadata = {
        'model_type': 'RandomForestClassifier',
        'preprocessing': 'RobustScaler + SMOTE-Tomek',
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'hyperparameters': MODEL_HYPERPARAMS,
        'training_date': training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration_seconds': training_duration,
        'data_split': {
            'train_samples': int(len(X_train)),
            'validation_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'resampled_train_samples': int(len(X_train_resampled))
        },
        'performance': results,
        'random_state': RANDOM_STATE
    }
    
    save_model_artifacts(
        clf, scaler, metadata,
        MODEL_OUTPUT_PATH, SCALER_OUTPUT_PATH, METADATA_OUTPUT_PATH, WEIGHTS_OUTPUT_PATH
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total training time: {training_duration:.2f} seconds")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {results['test']['accuracy']:.4f}")
    print(f"  F1 Score (Weighted): {results['test']['f1_weighted']:.4f}")
    print(f"  F1 Score (Macro): {results['test']['f1_macro']:.4f}")
    print("\nModel ready for deployment!")
    print("=" * 60)


if __name__ == "__main__":
    main()
