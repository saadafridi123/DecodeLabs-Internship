
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Core Framework Dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_and_inspect_data(file_path):
    """
    Ingests the transactional dataset from a CSV representation and generates
    critical structural diagnostics.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file not found at path: {file_path}")
        
    print("[INFO] Loading E-Commerce Transaction Dataset...")
    df = pd.read_csv(file_path)
    
    # Structural Diagnostics
    print(f"\n[DIAGNOSTIC] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[DIAGNOSTIC] Feature Data Types & Column Index:")
    print(df.dtypes)
    
    # Check Target Distribution Balance
    if 'OrderStatus' in df.columns:
        print("\n[DIAGNOSTIC] Target Variable 'OrderStatus' Distribution:")
        print(df['OrderStatus'].value_counts(normalize=True) * 100)
    else:
        raise KeyError("Crucial target column 'OrderStatus' missing from dataset.")
        
    return df

def partition_dataset(df, target_column='OrderStatus'):
    """
    Isolates label matrices from feature dataframes, explicitly drops operational
    or high-cardinality keys to prevent model overfitting, and executes an 
    80/20 stratified split.
    """
    # High-cardinality or row-identifier columns that cause extreme overfitting 
    # and do not contain predictive signals are filtered out safely.
    cols_to_drop = [target_column, 'OrderID', 'Date', 'CustomerId', 'TrackingNum', 'ShippingAddr']
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=existing_drops)
    y = df[target_column]
    
    # Enforcing Stratified Split to guarantee training and testing sets maintain
    # identical proportions of outcome classes (e.g., Delivered, Cancelled, Returned)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )
    
    print(f"\n[INFO] Stratified Data Partitioning Complete.")
    print(f"       -> Training Feature Set Shape: {X_train.shape}")
    print(f"       -> Testing Feature Set Shape:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_and_train_pipeline(X_train, y_train):
    """
    Constructs isolated processing workflows for numerical vs categorical variables
    and bundles them with an Ensemble Estimator into an atomic, production pipeline.
    """
    # Dynamic column grouping based on typical dataset attributes
    numerical_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    
    print(f"\n[INFO] Generating Modular Preprocessing Sub-Pipelines...")
    print(f"       Numerical Features Processing ({len(numerical_features)}): {numerical_features}")
    print(f"       Categorical Features Processing ({len(categorical_features)}): {categorical_features}")

    # Numerical Transformer: Re-scales to zero mean and unit variance
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: One-hot encodes nominal values with absolute safety against unseen test tags
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Consolidating sub-transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Bundling preprocessing blocks and the Random Forest classification engine into a single pipeline
    # class_weight='balanced' handles any intrinsic class discrepancies cleanly.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    print("\n[MODELING] Optimizing Random Forest Classifier Pipeline parameters...")
    pipeline.fit(X_train, y_train)
    print("[MODELING] Optimization complete. Pipeline successfully finalized for deployment.")
    
    return pipeline

def evaluate_pipeline_performance(model_pipeline, X_test, y_test):
    """
    Evaluates raw testing datasets directly via the pipeline, calculating precision, 
    recall, F1-scores, and mapping out a true validation Confusion Matrix.
    """
    # Inference generation
    y_pred = model_pipeline.predict(X_test)
    
    # 1. Classification Metrics Summary
    print("\n=================== PRODUCTION PERFORMANCE REPORT ===================")
    print(classification_report(y_test, y_pred))
    
    # 2. Overall Accuracy Metric
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Aggregated Pipeline Accuracy Status: {overall_accuracy * 100:.2f}%\n")
    
    # 3. Validation Matrix Plot Visualization
    print("[INFO] Rendering Visual Confusion Matrix Dashboard...")
    cm = confusion_matrix(y_test, y_pred)
    classes = model_pipeline.classes_
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style='whitegrid')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes,
                cbar=True,
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title('Validation Set Confusion Matrix Evaluation', fontsize=14, pad=15)
    plt.ylabel('Observed Reality (True Label)', fontsize=12)
    plt.xlabel('Algorithm Decision (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Configuration - Replace with the actual clean CSV path generated from your Sheet1 file
    DATA_PATH = "Dataset_for_Data_Analytics.csv" 
    
    # Mock data creator placeholder - allows standalone testing of the complete script instantly
    if not os.path.exists(DATA_PATH):
        print(f"[SYSTEM] '{DATA_PATH}' not found in current directory. Fabricating sample transactional template for script runtime testing...")
        mock_data = pd.DataFrame({
            'OrderID': [f'ORD{i}' for i in range(20000, 20100)],
            'Date': ['2024-05-12'] * 100,
            'CustomerId': [f'C{np.random.randint(10000,99999)}' for _ in range(1000)],
            'Product': np.random.choice(['Monitor', 'Phone', 'Tablet', 'Chair', 'Printer', 'Desk'], size=100),
            'Quantity': np.random.randint(1, 10, size=100),
            'UnitPrice': np.random.uniform(50.0, 800.0, size=100),
            'ShippingAddr': ['Main St'] * 100,
            'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Online', 'Cash', 'Gift Card'], size=100),
            'OrderStatus': np.random.choice(['Delivered', 'Cancelled', 'Returned', 'Shipped', 'Pending'], size=100),
            'TrackingNum': [f'TRK{np.random.randint(100000,999999)}' for _ in range(100)],
            'ItemsInCart': np.random.randint(1, 12, size=100),
            'CouponCode': np.random.choice(['SAVE10', 'WINTER15', 'FREESHIP', 'None'], size=100),
            'ReferralSource': np.random.choice(['Instagram', 'Facebook', 'Email', 'Referral'], size=100),
            'TotalPrice': np.random.uniform(100.0, 3000.0, size=100)
        })
        mock_data.to_csv(DATA_PATH, index=False)

    try:
        # Step 1: Run Data Quality Ingestion
        dataset = load_and_inspect_data(DATA_PATH)
        
        # Step 2: Split cohorts safely avoiding any data exposure / leakages
        X_train, X_test, y_train, y_test = partition_dataset(dataset)
        
        # Step 3 & 4: Build processing pipeline and fit model parameters
        trained_system_pipeline = build_and_train_pipeline(X_train, y_train)
        
        # Step 5: Evaluate outcomes and generate analytics visualizations
        evaluate_pipeline_performance(trained_system_pipeline, X_test, y_test)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution stopped: {str(e)}")