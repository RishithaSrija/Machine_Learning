import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

#page configuration

st.set_page_config("Simple Linear Regression",layout='centered')

#Load css

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}",unsafe_allow_html=True)
load_css("style.css")

# Title

st.markdown(""" 
<div class="card">
            <h1>Linear Regression</h1>
            <p>Predict <b>Tip amount </b> based on <b> Total bill</b> using Linear Regression... </p>

</div>
""",unsafe_allow_html=True)

#Load data
@st.cache_data

def load_data():
    return sns.load_dataset('tips')
df=load_data()

#Data preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Data Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

#Prepare data

x,y=df[['total_bill']],df['tip']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Train model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test) -1) / (len(y_test)-2)

#Visualization

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df['tip'],alpha=0.6)
ax.plot(df['total_bill'],model.predict(scaler.transform(x)),color='red')
ax.set_xlabel("Total Bill ($) ")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#Performance
st.markdown('<div class ="card"',unsafe_allow_html=True)
st.subheader('Model Performance ')
c1,c2=st.columns(2)
c1.metric("MAE ",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R2",f"{r2: .3f}")
c4.metric("Adj R2",f"{adj_r2: .3f}" )
st.markdown('</div>',unsafe_allow_html=True)

st.markdown(f""" 
        <div class="card">
        <h3>Nodel Interception</h3>
        <p><b>co-efficient:</b> {model.coef_[0]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}</p>
        </div>

""",unsafe_allow_html=True)

#Predictions
st.markdown('<div class ="card">',unsafe_allow_html=True)
st.subheader("Predicted Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    min_value=float(df['total_bill'].min()),
    max_value=float(df['total_bill'].max()),
    value=30.0
)

tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box" > Predicted Tip : ${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)

