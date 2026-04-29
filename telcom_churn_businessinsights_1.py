import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, precision_score, recall_score 

import seaborn as sns
import matplotlib.pyplot as plt

st.title('📊 Random Forest - Telecom Customer Churn System')

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(r'C:\Users\ADMIN\Desktop\abdul\Kaggle DataSets for DS & AI Ml Project\Telco_Customer_Churn.xlsx')

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    return df

df = load_data()
# -----------------------------
# SHOW DATA
# -----------------------------
st.subheader("Telecom Customer Churn Dataset")
if st.checkbox("Click to View Data"):
    st.write(df.head())
# -----------------------------
# ENCODING (same style)
# -----------------------------
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = le.fit_transform(df_encoded[col])

# -----------------------------
# SPLIT
# -----------------------------
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
random_state = st.number_input('Random State', 0, 100, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# -----------------------------
# MODEL PARAMETERS
# -----------------------------
n_estimators = st.slider('Number of Trees', 10, 200, 100)# use no of trees = 130 
max_depth = st.slider('Max Depth', 1, 20, 5)# use max depth = 6
bootstrap=st.selectbox('Bootstrap samples when building trees',[True,False])

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    bootstrap=bootstrap,
    class_weight='balanced',
    random_state=random_state)

model.fit(X_train, y_train)
threshold = st.slider('Churn Probability Threshold', 0.0, 1.0, 0.5)# use threshold = 0.58
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test) if threshold == 0.5 else (y_prob >= threshold).astype(int)

# -----------------------------
# EVALUATION
# -----------------------------
st.subheader('📊 Model Evaluation')

acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
st.write(f'✅ Accuracy: {acc:.3f}')
st.write(f'📉 MSE: {mse:.3f}')
st.write(f'📊 Precision: {precision:.2f}')
st.write(f'📈 Recall: {recall:.2f}')

st.text('Classification Report:')
st.text(classification_report(y_test, y_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.subheader('📌 Confusion Matrix')

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')

st.pyplot(fig)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.subheader("🔥 Important Features")

importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.write(feat_df.head(10))

# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------
st.subheader("📊 Business Insights")

st.write("Churn based on Contract:")
st.write(df.groupby("Contract")["Churn"].value_counts(normalize=True))

st.write("Churn based on Tech Support:")
st.write(df.groupby("TechSupport")["Churn"].value_counts(normalize=True))

st.write("💡 Insight: Month-to-month & No TechSupport customers tend to churn more")


# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🔮 Predict Customer Churn")

input_data = []

for col in X.columns:
    val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_data.append(val)

# -----------------------------
# BUSINESS RECOMMENDATION PREDICTION
# -----------------------------
st.subheader("🎯 Business Recommendation Prediction")

input_dict = dict(zip(X.columns, input_data))
contract_val = input_dict.get("Contract", None)
tech_val = input_dict.get("TechSupport", None)
monthly_val = input_dict.get("MonthlyCharges", None)

col1, col2 = st.columns(2)

with col1:
    st.write("Displaying Risk Factors Based On : **Contract | TechSupport | MonthlyCharges**")
    
    # Calculate risk score based on business factors
    risk_score = 0
    risk_factors = []
    
    if contract_val == 0:  # Month-to-month
        risk_score += 2
        risk_factors.append("Month-to-month contract")

    if tech_val == 0:  # No tech support
        risk_score += 0.25
        risk_factors.append("No Tech Support")

    if monthly_val > 80:
        risk_score += 0.25
        risk_factors.append("High Monthly Charges")
    
    # Additional risk factors from the model
    business_prob = min(risk_score, 1.0)
    
    st.write(f"\n**Risk Factors Identified: {len(risk_factors)}**")
    for factor in risk_factors:  
        st.write(f" • {factor}")  

with col2:
    if st.button("🔮 Predict from Business Recommendations"):
        # Use model prediction for more accurate result
        model_prediction = model.predict([input_data])[0]
        model_prob = model.predict_proba([input_data])[0][1]
        
        # Combine model prediction with business logic
        combined_prob = (model_prob + business_prob) / 2
        
        st.write("---")
        st.write("**Prediction Result:**")
        
        if model_prediction == 1:
            st.error("❌ Will Churn")
        else:
            st.success("✅ Will Not Churn")
       
        st.write(f"**Model Probability:** {model_prob:.2%}")
        st.write(f"**Business Risk Score:** {business_prob:.2%}")
        st.write(f"**Combined Score:** {combined_prob:.2%}")
        
        # Recommendation based on combined analysis
        if combined_prob > 0.5:
            st.warning("⚠️ High Risk - Action Required!")
        elif combined_prob > 0.3:
            st.info("⚡ Medium Risk - Monitor Closely")
        else:
            st.success("✅ Low Risk - Maintain Engagement")
    #In streamlit Provide the sliders with these nos to get good Accuracy, Precision,
    # and Confusion matrix values.
        # use no of trees = 130 
        # use max depth = 6
        # use threshold = 0.58