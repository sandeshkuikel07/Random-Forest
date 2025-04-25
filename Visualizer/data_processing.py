import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

def preprocess_data(df, selected_features):
    """
    Preprocess the data for modeling.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    selected_features : list
        List of features to use for modeling
        
    Returns:
    --------
    X : array-like
        Preprocessed features
    y : array-like
        Target values
    """
    # Extract features and target
    X = df[selected_features].values
    y = df['diagnosis'].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target values
    y_test : array-like
        Testing target values
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
