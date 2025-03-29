#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Make sure plots display in the notebook or IDE
import plotly.io as pio
pio.renderers.default = "browser"  # This will open plots in your default browser

# Step 1: Load and Analyze Data
file_path = r"C:\Users\Administrator\Downloads\Data analyis\security_incidents.csv"
data = pd.read_csv(file_path)

# Clean data and create features
data = data.drop_duplicates()
data['Casualty Ratio'] = data['Total killed'] / (data['Total affected'] + 1e-5)
data['Log_Total_Affected'] = np.log1p(data['Total affected'])

# Handle missing values
for col in data.select_dtypes(include=['number']).columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Step 2: Interactive Data Visualization
def plot_correlation_heatmap():
    """Generate and save an interactive correlation heatmap"""
    corr = data.corr(numeric_only=True)
    fig = px.imshow(corr, 
                    text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlations")
    fig.show()  # This will open in browser

def plot_histograms():
    """Generate and save interactive histograms for key numerical columns"""
    numerical_cols = ['Year', 'Month', 'Total killed', 'Total wounded', 'Total affected']
    for col in numerical_cols:
        fig = px.histogram(data, x=col, title=f"Distribution of {col}")
        fig.show()  # This will open in browser

def plot_scatter_matrix():
    """Generate and save an interactive scatter matrix"""
    fig = px.scatter_matrix(
        data, 
        dimensions=['Year', 'Month', 'Total killed', 'Total wounded', 'Total affected'],
        title="Relationships Between Features"
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()  # This will open in browser

def plot_boxplots():
    """Generate and save interactive boxplots"""
    fig = go.Figure()
    for col in ['Total killed', 'Total wounded', 'Total affected']:
        fig.add_trace(go.Box(y=data[col], name=col))
    fig.update_layout(title="Outlier Detection with Boxplots")
    fig.show()  # This will open in browser

# Step 3: Train & Evaluate Regression Model
def linear_regression_example():
    """Run linear regression analysis with interactive visualizations"""
    numerical_features = ['Year', 'Month', 'Day', 'Total killed', 'Total wounded', 'Casualty Ratio', 'Log_Total_Affected']
    target = 'Total affected'
    
    if target not in data.columns:
        print(f"Required target column not found. Available columns: {data.columns.tolist()}")
        return
    
    X = data[numerical_features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = model.score(X_test, y_test)
    print(f"Linear Regression MSE: {mse}")
    print(f"R-squared: {r2}")
    
    # Interactive Residual Plot
    residuals = y_test - predictions
    fig = px.scatter(
        x=predictions, 
        y=residuals,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        title="Residual Plot for Regression"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.show()  # This will open in browser
    
    # Feature Importance
    coef_df = pd.DataFrame({
        'Feature': numerical_features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    fig = px.bar(
        coef_df, 
        x='Feature', 
        y='Coefficient',
        title="Feature Importance in Linear Regression"
    )
    fig.show()  # This will open in browser

# Step 4: Logistic Regression
def logistic_regression_example():
    """Run logistic regression analysis with interactive visualizations"""
    numerical_features = ['Year', 'Month', 'Day', 'Total killed', 'Total wounded', 'Casualty Ratio', 'Log_Total_Affected']
    label = 'Verified'
    
    if label not in data.columns:
        print(f"Required label column not found. Available columns: {data.columns.tolist()}")
        return
    
    # Create a copy of data to avoid modifying the original
    data_copy = data.copy()
    
    # Convert 'Verified' to binary values
    data_copy[label] = data_copy[label].astype(str).str.lower()
    data_copy[label] = data_copy[label].map({'yes': 1, 'no': 0})
    
    if data_copy[label].isnull().any():
        print("Warning: Unmapped values found in 'Verified' column.")
        return
    
    X = data_copy[numerical_features]
    y = data_copy[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, predictions)
    proba = model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"Logistic Regression Accuracy: {acc}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"ROC AUC Score: {roc_auc}")
    
    # Interactive ROC Curve
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.show()  # This will open in browser
    
    # Interactive Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        title="Confusion Matrix"
    )
    fig.show()  # This will open in browser

# Main function to run the analysis
def run_analysis():
    print("Data preview:")
    print(data.head())
    
    print("\nGenerating interactive plots - they will open in your browser...")
    
    # Basic visualizations
    plot_correlation_heatmap()
    plot_histograms()
    plot_scatter_matrix()
    plot_boxplots()
    
    # ML models
    print("\nRunning regression analysis...")
    linear_regression_example()
    
    print("\nRunning classification analysis...")
    logistic_regression_example()
    
    print("\nAnalysis complete! All plots should have opened in your browser.")

# Execute the analysis
if __name__ == "__main__":
    run_analysis()


# In[16]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Load the data file
file_path = r"C:\Users\Administrator\Downloads\Data analyis\security_incidents.csv"
data = pd.read_csv(file_path)

def preprocess_data(data):
    """
    Comprehensive data preprocessing function
    
    Performs:
    - Duplicate removal
    - Feature engineering
    - Missing value handling
    - Verification of target variable
    """
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Feature engineering
    data['Casualty Ratio'] = data['Total killed'] / (data['Total affected'] + 1e-5)
    data['Log_Total_Affected'] = np.log1p(data['Total affected'])
    
    # Handle missing values in numerical columns
    for col in data.select_dtypes(include=['number']).columns:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Handle missing values in categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Detailed preprocessing of 'Verified' column
    print("\n--- Data Verification ---")
    print("Original 'Verified' column values:", data['Verified'].unique())
    
    # Create a custom mapping for verification status
    verification_mapping = {
        'Yes': 1,      # Explicitly verified
        'Pending': 0,  # Not yet verified
        'Pen': 0,      # Likely means Pending
        'Archived': 0  # Possibly not actively verified
    }
    
    # Apply mapping
    data['Verified_Binary'] = data['Verified'].map(verification_mapping)
    
    # Verify the new binary column
    print("\nVerified Binary Column:")
    print("Unique values:", data['Verified_Binary'].unique())
    print("Class distribution:")
    print(data['Verified_Binary'].value_counts(normalize=True))
    
    return data

def random_forest_classification(data):
    """
    Perform Random Forest Classification with Enhanced Error Handling
    
    Provides comprehensive diagnostic information about the classification process
    """
    # Feature selection
    numerical_features = [
        'Year', 'Month', 'Day', 
        'Total killed', 'Total wounded', 
        'Casualty Ratio', 'Log_Total_Affected'
    ]
    
    label = 'Verified_Binary'
    
    # Prepare data
    data_copy = data.copy()
    
    # Check class distribution
    class_dist = data_copy[label].value_counts(normalize=True)
    print("\nClass Distribution:")
    print(class_dist)
    
    # Prepare features and target
    X = data_copy[numerical_features]
    y = data_copy[label]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,  
        max_depth=None,    
        min_samples_split=2, 
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Train the model
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test_scaled)
    y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]
    
    # Performance Metrics
    print("\n--- Classification Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Visualization of Feature Importance
    importance = permutation_importance(rf_classifier, X_test_scaled, y_test, n_repeats=10, random_state=42)
    feature_importance = pd.DataFrame({
        'feature': numerical_features,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plotly visualizations
    fig_importance = px.bar(
        feature_importance, 
        x='feature', 
        y='importance', 
        title='Random Forest Feature Importance'
    )
    fig_importance.show()
    
    # ROC Curve Visualization
    roc_fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    )
    roc_fig.add_shape(
        type='line', 
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    roc_fig.show()

# Main execution
if __name__ == "__main__":
    preprocessed_data = preprocess_data(data)
    random_forest_classification(preprocessed_data)


# In[ ]:




