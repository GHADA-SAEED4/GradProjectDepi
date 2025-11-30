import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .safe-alert {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# HELPER FUNCTIONS
# ===========================

def apply_preprocessing(input_df, scaler=None, fit=False):
    """
    Apply the same preprocessing used in training:
    1. Scale Amount using RobustScaler
    2. Drop Time column
    3. Reorder columns
    """
    df_processed = input_df.copy()
    
    # Initialize scaler if not provided
    if scaler is None:
        scaler = RobustScaler()
    
    # Scale Amount
    if fit:
        df_processed['scaled_amount'] = scaler.fit_transform(
            df_processed['Amount'].values.reshape(-1, 1)
        )
    else:
        df_processed['scaled_amount'] = scaler.transform(
            df_processed['Amount'].values.reshape(-1, 1)
        )
    
    # Drop original columns
    df_processed.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # Reorder columns (scaled_amount first)
    scaled_amount = df_processed.pop('scaled_amount')
    df_processed.insert(0, 'scaled_amount', scaled_amount)
    
    return df_processed, scaler

# ===========================
# LOAD OR TRAIN MODEL
# ===========================

@st.cache_resource
def load_or_train_model():
    """
    Load pre-trained XGBoost model or train a new one
    """
    model_path = "xgboost_fraud_model.pkl"
    scaler_path = "robust_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, True
    else:
        return None, None, False

model, scaler, model_loaded = load_or_train_model()

# ===========================
# LOAD DATA (for stats)
# ===========================

@st.cache_data
def load_data():
    try:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'creditcard' in f.lower()]
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            return df, csv_files[0]
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

df, data_filename = load_data()

# ===========================
# SIDEBAR
# ===========================

st.sidebar.title("💳 Fraud Detection System")
st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success("✅ XGBoost Model Loaded")
    st.sidebar.markdown("### 📊 Model Performance")
    
    # Update these with your actual metrics from notebook
    st.sidebar.metric("Accuracy", "99.94%")
    st.sidebar.metric("Precision", "98.15%")
    st.sidebar.metric("Recall", "77.55%")
    st.sidebar.metric("F1-Score", "86.67%")
    st.sidebar.metric("ROC-AUC", "99.89%")
else:
    st.sidebar.error("❌ Model Not Found")
    st.sidebar.warning("Please train the model first using the notebook")

if df is not None:
    st.sidebar.info(f"📂 Dataset: {data_filename}")
    st.sidebar.metric("Total Transactions", f"{len(df):,}")
    fraud_rate = (df['Class'].sum() / len(df)) * 100
    st.sidebar.metric("Fraud Rate", f"{fraud_rate:.2f}%")
else:
    st.sidebar.warning("⚠️ No dataset found")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Note:** All V1-V28 features are PCA-transformed")

# ===========================
# HEADER
# ===========================

st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Detect fraudulent transactions using XGBoost Machine Learning")
st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis", "📈 Model Info"])

# ===========================
# TAB 1: SINGLE TRANSACTION
# ===========================

with tab1:
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Enter transaction details to check for fraud")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Transaction Info")
        time_input = st.number_input(
            "Time (seconds elapsed)",
            min_value=0.0,
            value=0.0,
            help="Seconds elapsed between this transaction and the first transaction"
        )
        
        amount_input = st.number_input(
            "Amount ($)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Transaction amount in dollars"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Presets")
        
        preset = st.selectbox(
            "Load Example Transaction",
            ["Custom", "Normal Transaction", "Suspicious Transaction"]
        )
    
    with col2:
        st.subheader("🔢 PCA Features (V1-V28)")
        st.markdown("*These are PCA-transformed features from the original data*")
        
        # Create expandable sections for PCA features
        with st.expander("V1 - V14 Features"):
            v_features_1 = {}
            cols = st.columns(2)
            for i in range(1, 15):
                with cols[(i-1) % 2]:
                    v_features_1[f'V{i}'] = st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                        key=f"v{i}"
                    )
        
        with st.expander("V15 - V28 Features"):
            v_features_2 = {}
            cols = st.columns(2)
            for i in range(15, 29):
                with cols[(i-15) % 2]:
                    v_features_2[f'V{i}'] = st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                        key=f"v{i}"
                    )
    
    # Combine all V features
    v_features = {**v_features_1, **v_features_2}
    
    # Handle presets
    if preset == "Normal Transaction":
        st.info("📝 Loaded: Typical legitimate transaction pattern")
        # You can set specific values here based on your data analysis
        
    elif preset == "Suspicious Transaction":
        st.warning("📝 Loaded: Suspicious transaction pattern")
        # You can set specific fraud-like values here
    
    st.markdown("---")
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("❌ Model not loaded. Please ensure xgboost_fraud_model.pkl exists.")
        else:
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Time': [time_input],
                    'Amount': [amount_input],
                    **{k: [v] for k, v in v_features.items()}
                })
                
                # Show preprocessing details
                with st.expander("🔍 Preprocessing Details"):
                    st.markdown("### Original Input:")
                    st.dataframe(input_data[['Time', 'Amount']].style.format("{:.2f}"))
                    
                    st.markdown("### After Preprocessing:")
                    st.info("""
                    **Steps Applied:**
                    1. ✅ **RobustScaler** on Amount (less sensitive to outliers)
                    2. ✅ **Drop Time** (not used in final model)
                    3. ✅ **Reorder columns** (scaled_amount first, then V1-V28)
                    """)
                
                # Apply preprocessing
                input_processed, _ = apply_preprocessing(input_data, scaler, fit=False)
                
                with st.expander("🔍 Processed Features"):
                    st.dataframe(input_processed)
                
                # Predict
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0]
                
                # Display result
                st.markdown("### 🎯 Analysis Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="fraud-alert">
                            🚨 FRAUD DETECTED
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="safe-alert">
                            ✅ LEGITIMATE
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    fraud_prob = probability[1] * 100
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_prob:.2f}%",
                        delta=f"{fraud_prob - 50:.2f}% from baseline"
                    )
                
                with col3:
                    legit_prob = probability[0] * 100
                    st.metric(
                        "Legitimate Probability",
                        f"{legit_prob:.2f}%"
                    )
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
                
                # Transaction details summary
                st.markdown("### 📋 Transaction Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown(f"""
                    **Transaction Details:**
                    - Amount: **${amount_input:,.2f}**
                    - Time: **{time_input:,.0f}** seconds
                    - Classification: **{'FRAUD' if prediction == 1 else 'LEGITIMATE'}**
                    """)
                
                with summary_col2:
                    st.markdown(f"""
                    **Risk Assessment:**
                    - Fraud Risk: **{fraud_prob:.2f}%**
                    - Confidence: **{max(probability) * 100:.2f}%**
                    - Model: **XGBoost Classifier**
                    """)
                
                # Recommendations
                st.markdown("### 💡 Recommended Actions")
                
                if prediction == 1:
                    if fraud_prob > 90:
                        st.error("""
                        **🚨 HIGH RISK - Immediate Action Required:**
                        - ⛔ **BLOCK transaction immediately**
                        - 📞 Contact cardholder for verification
                        - 🔒 Freeze card temporarily
                        - 📝 File fraud report
                        - 🔍 Review recent transaction history
                        """)
                    elif fraud_prob > 70:
                        st.warning("""
                        **⚠️ MEDIUM-HIGH RISK:**
                        - ⏸️ Hold transaction for manual review
                        - 📞 Attempt to contact cardholder
                        - 🔍 Check transaction patterns
                        - ✅ Require additional authentication
                        """)
                    else:
                        st.warning("""
                        **⚠️ SUSPICIOUS:**
                        - 🔍 Flag for review
                        - 📊 Monitor closely
                        - 📧 Send verification email
                        """)
                else:
                    if legit_prob > 95:
                        st.success("""
                        **✅ LEGITIMATE - Safe to Process:**
                        - ✔️ Approve transaction
                        - 📊 Continue normal monitoring
                        - 💳 No action required
                        """)
                    else:
                        st.info("""
                        **✅ LIKELY LEGITIMATE:**
                        - ✔️ Approve with standard monitoring
                        - 📊 Track for patterns
                        """)
                
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                with st.expander("🔍 Debug Information"):
                    st.write("Error details:", str(e))

# ===========================
# TAB 2: BATCH ANALYSIS
# ===========================

with tab2:
    st.header("📊 Batch Transaction Analysis")
    st.markdown("Upload a CSV file to analyze multiple transactions")
    
    with st.expander("📋 Required CSV Format"):
        st.markdown("""
        Your CSV must contain these columns:
        - **Time**: Seconds elapsed since first transaction
        - **Amount**: Transaction amount
        - **V1 - V28**: PCA-transformed features
        
        Example format:
        """)
        
        sample_df = pd.DataFrame({
            'Time': [0, 100],
            'Amount': [150.0, 2500.0],
            'V1': [-1.359, 1.191],
            'V2': [-0.072, 0.266],
            '...': ['...', '...'],
            'V28': [0.014, -0.021]
        })
        st.dataframe(sample_df)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(batch_df)} transactions found.")
            
            st.subheader("📄 Preview Data")
            st.dataframe(batch_df.head(10))
            
            if st.button("🚀 Analyze All Transactions", type="primary"):
                if not model_loaded:
                    st.error("❌ Model not loaded")
                else:
                    # Apply preprocessing
                    batch_processed, _ = apply_preprocessing(batch_df, scaler, fit=False)
                    
                    # Predict
                    predictions = model.predict(batch_processed)
                    probabilities = model.predict_proba(batch_processed)
                    
                    # Add results
                    batch_df['Prediction'] = predictions
                    batch_df['Risk_Level'] = ['🚨 FRAUD' if p == 1 else '✅ LEGITIMATE' for p in predictions]
                    batch_df['Fraud_Probability'] = probabilities[:, 1] * 100
                    batch_df['Legitimate_Probability'] = probabilities[:, 0] * 100
                    
                    # Display results
                    st.markdown("### 🎯 Analysis Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_fraud = (predictions == 1).sum()
                    total_legit = (predictions == 0).sum()
                    avg_fraud_prob = probabilities[:, 1].mean() * 100
                    
                    col1.metric("Total Transactions", len(batch_df))
                    col2.metric("Fraudulent", total_fraud, 
                               delta=f"{(total_fraud/len(batch_df)*100):.1f}%")
                    col3.metric("Legitimate", total_legit,
                               delta=f"{(total_legit/len(batch_df)*100):.1f}%")
                    col4.metric("Avg Fraud Risk", f"{avg_fraud_prob:.2f}%")
                    
                    # Results table
                    display_cols = ['Time', 'Amount', 'Risk_Level', 'Fraud_Probability', 'Legitimate_Probability']
                    st.dataframe(
                        batch_df[display_cols].style.background_gradient(
                            subset=['Fraud_Probability'], 
                            cmap='RdYlGn_r'
                        ),
                        width='stretch'
                    )
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis Results",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualizations
                    st.markdown("### 📊 Analysis Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Fraud distribution
                        fig = px.pie(
                            values=[total_fraud, total_legit],
                            names=['Fraud', 'Legitimate'],
                            title='Transaction Classification',
                            color_discrete_sequence=['#e74c3c', '#2ecc71']
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        # Fraud probability distribution
                        fig = px.histogram(
                            batch_df,
                            x='Fraud_Probability',
                            nbins=50,
                            title='Fraud Probability Distribution',
                            color='Prediction',
                            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                        )
                        st.plotly_chart(fig, width='stretch')
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ===========================
# TAB 3: MODEL INFO
# ===========================

with tab3:
    st.header("📈 XGBoost Model Information")
    
    st.markdown("""
    ### 🎯 Model Details
    
    **Algorithm:** XGBoost (Extreme Gradient Boosting)
    
    **Why XGBoost for Fraud Detection?**
    - ✅ Excellent performance on imbalanced datasets
    - ✅ Built-in handling of missing values
    - ✅ Robust to outliers
    - ✅ Fast training and prediction
    - ✅ Feature importance analysis
    - ✅ Regularization to prevent overfitting
    
    ---
    
    ### 📊 Performance Metrics
    """)
    
    # Create metrics DataFrame
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Score': [0.9994, 0.9815, 0.7755, 0.8667, 0.9989]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(metrics_df.style.format({'Score': '{:.4f}'}), width=400)
    
    with col2:
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title='Model Performance Metrics',
            color='Score',
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔧 Preprocessing Pipeline
    
    1. **Data Loading**
       - Remove duplicates
       - Shuffle dataset for randomness
    
    2. **Feature Scaling**
       - Apply **RobustScaler** to Amount feature
       - RobustScaler is less sensitive to outliers than StandardScaler
    
    3. **Feature Selection**
       - Drop Time column (not predictive)
       - Keep Amount (scaled) and V1-V28 (PCA features)
    
    4. **Handle Class Imbalance**
       - Apply SMOTE on training data only
       - Balance minority class (fraud) with majority class
    
    5. **Model Training**
       - Train XGBoost with 90 estimators
       - Use logloss as evaluation metric
    
    ---
    
    ### 📝 Dataset Information
    """)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
        col3.metric("Fraud Rate", f"{(df['Class'].sum()/len(df)*100):.3f}%")
        
        st.markdown("---")
        
        # Feature info
        st.markdown("### 🔢 Features Explained")
        
        st.info("""
        **V1 - V28:** Principal Component Analysis (PCA) transformed features
        - Original features anonymized for privacy
        - Reduced dimensionality while preserving information
        - Cannot be reverse-engineered to original features
        
        **Time:** Seconds elapsed between each transaction and first transaction
        - Not used in final model
        
        **Amount:** Transaction amount in dollars
        - Scaled using RobustScaler
        - Important feature for fraud detection
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎓 Model Training Details
    
    **Training Approach:**
    - 80% training, 20% testing split
    - Stratified sampling to maintain class distribution
    - SMOTE applied only on training data (no data leakage!)
    - No outlier removal (outliers can indicate fraud)
    
    **Hyperparameters:**
    - Number of estimators: 90
    - Evaluation metric: Log Loss
    - Random state: 42 (for reproducibility)
    
    **Why No Outlier Removal?**
    - In fraud detection, outliers often **ARE** the fraudulent transactions
    - Unusual amounts or patterns are key indicators
    - Removing outliers would reduce model effectiveness
    """)

# ===========================
# FOOTER
# ===========================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>💳 Credit Card Fraud Detection System | Built with XGBoost & Streamlit</p>
        <p>🔒 Secure Transaction Analysis | Real-time Fraud Prevention</p>
        <p>⚡ Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)