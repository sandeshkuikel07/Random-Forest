# Breast Cancer Prediction with Random Forest

## 📋 Overview

This interactive dashboard helps visualize and interpret a Random Forest model trained to predict breast cancer diagnosis based on cell nucleus measurements. The model classifies tumors as either Malignant (M) or Benign (B) using features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

The application is built with Streamlit and provides an intuitive interface for exploring the Wisconsin Breast Cancer dataset, training Random Forest models with customizable parameters, and making predictions.

## 🔬 Features

- **Data Overview**: Explore dataset statistics, class distribution, feature histograms, and correlations
- **Model Training**: Train Random Forest models with customizable parameters
- **Model Performance**: Evaluate model performance with confusion matrix, ROC curve, and precision-recall curve
- **Interactive Predictions**: Input feature values and get real-time predictions
- **Decision Tree Visualization**: Visualize individual decision trees from the Random Forest

## 🔧 Technical Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms and tools
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Joblib**: Model serialization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sandeshkuikel07/Random-Forest.git

```


2. Run the Streamlit app:
```bash
streamlit run app.py
```


## 📊 Dataset Features

The dataset contains the following features:

- **Radius**: Mean of distances from center to points on the perimeter
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**: Perimeter of the cell nucleus
- **Area**: Area of the cell nucleus
- **Smoothness**: Local variation in radius lengths
- **Compactness**: Perimeter² / area - 1.0
- **Concavity**: Severity of concave portions of the contour
- **Concave points**: Number of concave portions of the contour
- **Symmetry**: Symmetry of the cell nucleus
- **Fractal dimension**: "Coastline approximation" - 1

Each feature has three variations:
- Mean value (`_mean`)
- Standard error (`_se`)
- Worst (mean of the three largest values) (`_worst`)

## 📝 Usage Guide

### Data Overview Tab
- View sample data and dataset information
- Explore class distribution via pie chart
- View feature distributions via histograms
- Analyze feature correlations via heatmap

### Model Training Tab
- Select features for model training
- Adjust model parameters:
  - Number of trees
  - Maximum depth
  - Minimum samples split
  - Minimum samples leaf
  - Maximum features
- View feature importance after training

### Model Performance Tab
- View confusion matrix
- Analyze classification report with precision, recall, and F1 scores
- Explore ROC curve and precision-recall curve
- Visualize 2D decision boundaries

### Make Predictions Tab
- Input feature values using sliders
- Get real-time predictions with probability gauge

### Decision Trees Tab
- Visualize individual decision trees from the Random Forest
- Understand decision paths and node distributions


## 🙏 Acknowledgments

- Scikit-learn for their excellent machine learning tools
- Streamlit for making interactive data apps easy to build
