import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------
# Load and prepare data
# -------------------------------

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

df = pd.read_csv("TelcoCustChurn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Use only few features
df = df[['tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'Churn']]

X = df.drop('Churn', axis=1)
y = df['Churn']

X = pd.get_dummies(X, drop_first=True)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train, y_train)

# -------------------------------
# Evaluation Metrics
# -------------------------------
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Simple Churn Prediction App")

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

st.subheader(" Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

st.subheader(" Confusion Matrix")
st.write(conf_matrix)


fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap='Blues',
    values_format='d',
    ax=ax
)
ax.set_title("Confusion Matrix")
st.pyplot(fig)


st.markdown("---")

st.subheader(" Predict Churn for New Customer")

tenure = st.number_input("Tenure (months)", 0, 72)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly = st.number_input("Monthly Charges", 0.0, 200.0)

input_df = pd.DataFrame([{
    "tenure": tenure,
    "Contract": contract,
    "InternetService": internet,
    "MonthlyCharges": monthly
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
