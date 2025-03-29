elif analysis_type == "Classification Analysis":
    st.header("Classification Models")
    
    try:
        # Enhanced preprocessing
        data_copy = data.copy()
        
        # Comprehensive verification mapping
        verification_mapping = {
            'yes': 1,
            'no': 0,
            'pending': 0,
            'pen': 0,
            'archived': 0,
            'unknown': 0,
            '': 0
        }
        
        data_copy['Verified'] = (
            data_copy['Verified']
            .astype(str)
            .str.lower()
            .str.strip()
            .map(verification_mapping)
            .fillna(0)  # Handle any unexpected values
            .astype(int)  # Convert to integers
        
        # Filter valid target values
        data_copy = data_copy[data_copy['Verified'].isin([0, 1])
        
        # Check class balance
        class_counts = data_copy['Verified'].value_counts()
        if len(class_counts) < 2:
            st.error("Not enough classes for classification (need at least 2 classes)")
            st.stop()
        
        # Model selection
        model_type = st.radio("Select Classifier", ["Logistic Regression", "Random Forest"])
        
        X = data_copy[['Year', 'Month', 'Day', 'Total killed', 'Total wounded', 'Casualty Ratio', 'Log_Total_Affected']]
        y = data_copy['Verified']
        
        # Final validation check
        if y.isnull().any():
            st.error("Target variable contains missing values after preprocessing!")
            st.stop()
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Maintain class balance
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)  # Increased max iterations
        else:
            model = RandomForestClassifier(
                class_weight='balanced',
                n_estimators=200,
                random_state=42
            )
        
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics display
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
        with col2:
            st.metric("ROC AUC", f"{auc(fpr, tpr):.2f}")  # Direct AUC calculation
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Classification failed: {str(e)}")
        st.stop()
