import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)

from model_utils import load_data, train_model, predict_single_sample
from visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve, 
    plot_decision_tree, plot_feature_importance, 
    plot_decision_boundaries, plot_histogram_features
)
from data_processing import preprocess_data, split_data

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction - Random Forest Dashboard",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title
st.title("🌲 Breast Cancer Prediction with Random Forest")

# Add a description
st.markdown("""
This interactive dashboard helps visualize and interpret a Random Forest model 
trained to predict breast cancer diagnosis based on cell nucleus measurements.

The model classifies tumors as either Malignant (M) or Benign (B) using features 
extracted from digitized images of fine needle aspirates (FNA) of breast masses.

### Dataset Features:
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
""")

# Load data
data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Loading data...done!')

# Sidebar for model parameters
st.sidebar.header("Model Parameters")

# Parameters selection
n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Max depth", 1, 30, 10, 1)
min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2, 1)
min_samples_leaf = st.sidebar.slider("Min samples leaf", 1, 20, 1, 1)
max_features = st.sidebar.select_slider(
    "Max features", 
    options=["sqrt", "log2", "auto", None],
    value="sqrt"
)

# Feature selection
st.sidebar.header("Feature Selection")
feature_selection = st.sidebar.multiselect(
    "Select features for visualization", 
    options=[col for col in df.columns if col not in ['id', 'diagnosis']],
    default=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean']
)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🧠 Model Training", 
    "📈 Model Performance", 
    "🔮 Make Predictions",
    "🌲 Decision Trees"
])

# Tab 1: Data Overview
with tab1:
    st.header("Breast Cancer Dataset Overview")
    
    # Show sample data
    st.subheader("Sample Data")
    st.write(df.head())
    
    # Data information
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of samples:** {df.shape[0]}")
        st.write(f"**Number of features:** {df.shape[1]-2}")  # Excluding 'id' and 'diagnosis'
    with col2:
        st.write("**Class distribution:**")
        class_dist = df['diagnosis'].value_counts().reset_index()
        class_dist.columns = ['Diagnosis', 'Count']
        class_dist['Diagnosis'] = class_dist['Diagnosis'].map({'M': 'Malignant', 'B': 'Benign'})
        fig = px.pie(class_dist, values='Count', names='Diagnosis', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature histograms
    st.subheader("Feature Distributions")
    if len(feature_selection) > 0:
        plot_histogram_features(df, feature_selection)
    else:
        st.warning("Please select features in the sidebar for visualization.")
    
    # Feature correlations
    st.subheader("Feature Correlations")
    if len(feature_selection) > 0:
        corr = df[feature_selection].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select features in the sidebar for visualization.")

# Tab 2: Model Training
with tab2:
    st.header("Model Training")
    
    # Select the features for training
    st.subheader("Feature Selection for Model Training")
    all_features = [col for col in df.columns if col not in ['id', 'diagnosis']]
    selected_features = st.multiselect(
        "Select features for model training", 
        options=all_features,
        default=all_features
    )
    
    # Test-train split
    test_size = st.slider("Test set size (%)", 10, 50, 20, 5) / 100
    random_state = st.slider("Random state", 0, 100, 42, 1)
    
    # Train button
    train_btn = st.button("Train Model")
    
    if train_btn:
        if len(selected_features) < 2:
            st.error("Please select at least 2 features for training.")
        else:
            # Train the model
            with st.spinner('Training model...'):
                X, y = preprocess_data(df, selected_features)
                X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
                
                model, y_pred, feature_importances = train_model(
                    X_train, X_test, y_train, 
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state
                )
                
                # Save model and data to session state
                st.session_state['model'] = model
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['feature_importances'] = feature_importances
                st.session_state['selected_features'] = selected_features
                
                st.success("Model trained successfully!")
                
                # Display feature importances
                st.subheader("Feature Importance")
                
                fig = px.bar(
                    x=feature_importances, 
                    y=selected_features, 
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    title='Feature Importance',
                    color=feature_importances,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Performance
with tab3:
    st.header("Model Performance")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' tab.")
    else:
        # Get data from session state
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(conf_matrix)
        st.pyplot(fig_cm)
        
        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(class_report).transpose()
        report_df = report_df.round(3)
        st.dataframe(report_df)
        
        # ROC Curve
        st.subheader("ROC Curve")
        y_proba = model.predict_proba(X_test)[:, 1]
        fig_roc = plot_roc_curve(y_test, y_proba)
        st.pyplot(fig_roc)
        
        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        fig_pr = plot_pr_curve(y_test, y_proba)
        st.pyplot(fig_pr)
        
        # Display decision boundaries if we have 2 features selected
        if len(st.session_state['selected_features']) >= 2:
            st.subheader("Decision Boundaries Visualization")
            
            # Feature selection for visualization
            if len(st.session_state['selected_features']) > 2:
                feature_options = st.session_state['selected_features']
                x_feature = st.selectbox("Select X axis feature", feature_options, index=0)
                y_feature = st.selectbox("Select Y axis feature", feature_options, index=1)
                
                feature1_idx = st.session_state['selected_features'].index(x_feature)
                feature2_idx = st.session_state['selected_features'].index(y_feature)
            else:
                feature1_idx = 0
                feature2_idx = 1
                x_feature = st.session_state['selected_features'][feature1_idx]
                y_feature = st.session_state['selected_features'][feature2_idx]
            
            # Plot decision boundaries
            X = np.column_stack((st.session_state['X_train'], st.session_state['y_train']))
            fig_boundary = plot_decision_boundaries(
                model, X, feature1_idx, feature2_idx, 
                feature_names=[x_feature, y_feature]
            )
            st.pyplot(fig_boundary)

# Tab 4: Make Predictions
with tab4:
    st.header("Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' tab.")
    else:
        st.subheader("Enter Feature Values for Prediction")
        
        # Create a form for input
        with st.form("prediction_form"):
            # Create columns to organize inputs
            cols = st.columns(3)
            
            # Get feature ranges for sliders
            feature_mins = df[st.session_state['selected_features']].min()
            feature_maxs = df[st.session_state['selected_features']].max()
            feature_means = df[st.session_state['selected_features']].mean()
            
            # Create a dictionary to store the feature values
            feature_values = {}
            
            # Create sliders for each feature
            for i, feature in enumerate(st.session_state['selected_features']):
                col_idx = i % 3
                feature_values[feature] = cols[col_idx].slider(
                    feature,
                    float(feature_mins[feature]),
                    float(feature_maxs[feature]),
                    float(feature_means[feature]),
                    key=f"feature_{feature}"
                )
            
            # Add a predict button
            predict_button = st.form_submit_button("Predict")
        
        # Make prediction when the button is clicked
        if predict_button:
            # Create a single sample from the input values
            X_sample = np.array([feature_values[f] for f in st.session_state['selected_features']]).reshape(1, -1)
            
            # Make a prediction
            prediction, prediction_proba = predict_single_sample(
                st.session_state['model'],
                X_sample
            )
            
            # Display the prediction
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("Prediction: Malignant (M)")
                else:
                    st.success("Prediction: Benign (B)")
            
            with col2:
                st.write(f"Probability of Malignant: {prediction_proba[0][1]:.4f}")
                st.write(f"Probability of Benign: {prediction_proba[0][0]:.4f}")
            
            # Display a gauge chart for the probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba[0][1],
                title={'text': "Probability of Malignant"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Decision Trees
with tab5:
    st.header("Individual Decision Trees from the Forest")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' tab.")
    else:
        # Select a tree from the forest
        tree_idx = st.slider("Select tree index", 0, n_estimators-1, 0, 1)
        
        # Plot the selected decision tree
        st.subheader(f"Decision Tree #{tree_idx+1}")
        with st.spinner("Rendering decision tree visualization..."):
            fig_tree = plot_decision_tree(
                st.session_state['model'], 
                tree_idx, 
                st.session_state['selected_features']
            )
            st.pyplot(fig_tree)
        
        st.info("""
        **Decision Tree Interpretation Guide:**
        
        - **Nodes** contain conditions on features
        - **Branches** represent the decision paths
        - **Leaf nodes** show the final prediction and sample counts
        - **Colors** indicate the majority class (darker = stronger prediction)
        - **Values** show the count of samples [Benign, Malignant]
        
        Decision trees split the feature space by asking questions like "Is feature X <= value Y?"
        """)

# Footer
st.markdown("---")
st.markdown("""
**Dashboard created with Streamlit** 
| Random Forest model implementation using Scikit-learn
| Data source: Breast Cancer Wisconsin (Diagnostic) Dataset
""")
