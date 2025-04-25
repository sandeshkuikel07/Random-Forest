import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, confusion_matrix

def plot_histogram_features(df, features, hue='diagnosis'):
    """
    Plot histograms for selected features with hue based on diagnosis.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    features : list
        List of features to plot
    hue : str, default='diagnosis'
        Column to use for coloring
    """
    # Create a figure with subplots based on the number of features
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows*4))
    axes = axes.flatten()
    
    # Map diagnosis back to descriptive labels for visualization
    df_plot = df.copy()
    if hue == 'diagnosis':
        df_plot[hue] = df_plot[hue].map({1: 'Malignant', 0: 'Benign'})
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.histplot(
                data=df_plot, 
                x=feature, 
                hue=hue, 
                kde=True, 
                ax=axes[i],
                palette=['#2ecc71', '#e74c3c'] if hue == 'diagnosis' else None
            )
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_confusion_matrix(conf_matrix):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    conf_matrix : array-like of shape (2, 2)
        Confusion matrix
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the confusion matrix plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=False,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])
    
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probabilities)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    return fig

def plot_pr_curve(y_true, y_score):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probabilities)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the Precision-Recall curve plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    return fig

def plot_decision_tree(model, tree_idx, feature_names):
    """
    Plot a single decision tree from the Random Forest.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    tree_idx : int
        Index of the tree to plot
    feature_names : list
        Names of the features
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the decision tree plot
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    tree = model.estimators_[tree_idx]
    
    plot_tree(
        tree, 
        feature_names=feature_names,
        class_names=['Benign', 'Malignant'],
        filled=True,
        ax=ax,
        fontsize=8,
        precision=2,
        proportion=True,
        rounded=True
    )
    
    ax.set_title(f"Decision Tree #{tree_idx+1} from Random Forest", fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importances, feature_names):
    """
    Plot feature importances.
    
    Parameters:
    -----------
    feature_importances : array-like
        Importance of each feature
    feature_names : list
        Names of the features
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the feature importance plot
    """
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot feature importances
    ax.bar(range(len(feature_importances)), feature_importances[indices], color='skyblue')
    ax.set_xticks(range(len(feature_importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    
    plt.tight_layout()
    return fig

def plot_decision_boundaries(model, X, feature1_idx, feature2_idx, feature_names=None, mesh_step_size=0.01):
    """
    Plot decision boundaries of the model for two selected features.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    X : array-like
        Dataset with features and target (last column)
    feature1_idx : int
        Index of the first feature
    feature2_idx : int
        Index of the second feature
    feature_names : list, default=None
        Names of the features
    mesh_step_size : float, default=0.01
        Step size for the mesh grid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the decision boundary plot
    """
    # Extract the two features and target
    X_plot = X[:, [feature1_idx, feature2_idx]]
    y_plot = X[:, -1]
    
    # Create a mesh grid
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )
    
    # Create a specialized model that only uses these two features
    # Train a new model with just the two selected features
    from sklearn.ensemble import RandomForestClassifier
    specialized_model = RandomForestClassifier(
        n_estimators=50,  # Use fewer trees for visualization
        max_depth=model.max_depth,
        random_state=42
    )
    specialized_model.fit(X_plot, y_plot)
    
    # Make predictions on the mesh grid with the specialized model
    Z = specialized_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    
    # Plot the data points
    scatter = ax.scatter(
        X_plot[:, 0], 
        X_plot[:, 1], 
        c=y_plot, 
        cmap='RdBu', 
        edgecolor='k', 
        s=60, 
        alpha=0.7
    )
    
    # Set labels and title
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel(f'Feature {feature1_idx}')
        ax.set_ylabel(f'Feature {feature2_idx}')
    
    ax.set_title('Decision Boundaries (2D Projection)')
    
    # Add a legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    plt.tight_layout()
    return fig
