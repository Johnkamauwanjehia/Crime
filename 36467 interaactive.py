import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Configure page settings
st.set_page_config(page_title="Security Analysis", layout="wide", page_icon="🔒")
st.title("Security Incidents Interactive Dashboard")

# ================== Data Loading & Preprocessing ==================
@st.cache_data
def load_and_preprocess(uploaded_file):
    """Load and preprocess data with caching"""
    data = pd.read_csv(uploaded_file)
    
    # Clean data
    data = data.drop_duplicates()
    
    # Feature engineering
    data['Casualty Ratio'] = data['Total killed'] / (data['Total affected'] + 1e-5)
    data['Log_Total_Affected'] = np.log1p(data['Total affected'])
    
    # Handle missing values
    for col in data.select_dtypes(include=['number']).columns:
        data[col].fillna(data[col].median(), inplace=True)
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    return data

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to begin analysis")
    st.stop()

data = load_and_preprocess(uploaded_file)

# ================== Sidebar Controls ==================
analysis_type = st.sidebar.selectbox("Select Analysis Type", [
    "Data Overview",
    "Interactive Visualizations",
    "Regression Analysis",
    "Classification Analysis"
])

# ================== Main Content ==================
if analysis_type == "Data Overview":
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First 5 Rows")
        st.write(data.head())
    with col2:
        st.subheader("Dataset Statistics")
        st.write(data.describe())
    
    st.subheader("Column Information")
    st.write(pd.DataFrame({
        'Column Name': data.columns,
        'Data Type': data.dtypes,
        'Missing Values': data.isnull().sum()
    }))

elif analysis_type == "Interactive Visualizations":
    st.header("Interactive Visualizations")
    
    viz_option = st.selectbox("Choose Visualization", [
        "Correlation Heatmap",
        "Feature Distributions",
        "Scatter Matrix",
        "Outlier Detection"
    ])
    
    if viz_option == "Correlation Heatmap":
        corr = data.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Feature Distributions":
        selected_col = st.selectbox("Select Feature", ['Year', 'Month', 'Total killed', 'Total wounded'])
        fig = px.histogram(data, x=selected_col, nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Scatter Matrix":
        fig = px.scatter_matrix(data, dimensions=['Year', 'Month', 'Total killed', 'Total wounded'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Outlier Detection":
        fig = go.Figure()
        for col in ['Total killed', 'Total wounded', 'Total affected']:
            fig.add_trace(go.Box(y=data[col], name=col))
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Regression Analysis":
    st.header("Linear Regression Analysis")
    
    # Model setup
    X = data[['Year', 'Month', 'Day', 'Total killed', 'Total wounded', 'Casualty Ratio', 'Log_Total_Affected']]
    y = data['Total affected']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mean_squared_error(y_test, predictions):.2f}")
    with col2:
        st.metric("R-squared Score", f"{model.score(X_test, y_test):.2f}")
    
    # Residual plot
    st.subheader("Residual Analysis")
    residuals = y_test - predictions
    fig = px.scatter(x=predictions, y=residuals, trendline="lowess")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Classification Analysis":
    st.header("Classification Models")
    
    # Preprocessing for classification
    data_copy = data.copy()
    data_copy['Verified'] = data_copy['Verified'].str.lower().map({'yes': 1, 'no': 0})
    
    # Model selection
    model_type = st.radio("Select Classifier", ["Logistic Regression", "Random Forest"])
    
    X = data_copy[['Year', 'Month', 'Day', 'Total killed', 'Total wounded', 'Casualty Ratio', 'Log_Total_Affected']]
    y = data_copy['Verified']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier(class_weight='balanced')
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics display
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
    with col2:
        st.metric("ROC AUC", f"{auc(roc_curve(y_test, proba)[0], roc_curve(y_test, proba)[1]):.2f}")
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig, use_container_width=True)
