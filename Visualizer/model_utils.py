import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

@st.cache_data
def load_data():
    """Load and prepare the breast cancer dataset."""
    try:
        df = pd.read_csv(r"C:\Users\ASUS\Desktop\AI project\breast-cancer.csv")
        # Replace diagnosis values with binary values for modeling
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def train_model(X_train, X_test, y_train, n_estimators=100, max_depth=None, 
                min_samples_split=2, min_samples_leaf=1, max_features='sqrt', 
                random_state=42):
    """
    Train a Random Forest model with the given parameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Target values for training
    n_estimators : int, default=100
        The number of trees in the forest
    max_depth : int or None, default=None
        The maximum depth of the tree
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node
    max_features : {'sqrt', 'log2', 'auto'} or None, default='sqrt'
        The number of features to consider when looking for the best split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained Random Forest model
    y_pred : array-like
        Predictions on test set
    feature_importances : array-like
        Importance of each feature
    """
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    return model, y_pred, feature_importances

def predict_single_sample(model, X_sample):
    """
    Make a prediction for a single sample.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    X_sample : array-like
        Feature values for the sample
        
    Returns:
    --------
    prediction : int
        Predicted class (0 for Benign, 1 for Malignant)
    prediction_proba : array-like
        Probabilities for each class
    """
    prediction = model.predict(X_sample)
    prediction_proba = model.predict_proba(X_sample)
    
    return prediction[0], prediction_proba
