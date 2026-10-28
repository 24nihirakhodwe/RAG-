import streamlit as st
import pandas as pd
import numpy as np 

st.title("hello streamlit")
st.write(":streamlit :This is your first streamlit app" )
st.text("lets get started")

## conditional logic with widgets 
name=st.text_input("enter your name:")
if st.button("greet"):
    st.success(f"hello,{name}!")

#displaying data and charts 
df=pd.DataFrame(np.random.randn(10,2),columns=["A","B"])
st.line_chart(df) 
st.bar_chart(df)


#file uploading and caching topics 
upload_file=st.file_uploader("upload a csv ", type="csv")
if upload_file:
    df=pd.read_csv(upload_file)
    st.dataframe(df)

st.title(" Text and Markdown Demo")
st.header("This is header")
st.subheader("This is subheader")
st.markdown("**Bold**,Italic*,'Code',[Link](https://streamlit.io)")
st.code("for i in range(5):print(i)",language="python")


st.text_input("what is your name?")
st.text_area("write something....")
st.number_input("pick a number",min_value=0,max_value=100)
st.slider("choose a range",0,100)
st.selectbox("select a fruit",["Apple","Banana","Mango"])
st.multiselect("choose toppings ",["paparika","olive","greenolive","cheese"])
st.radio("Pick one",["Option A","Option B "])
st.checkbox("I agree to the terms ")


if st.checkbox("show details "):
    st.info("here are more details.....")


option=st.radio("choose a view",["show chart","show table"])
if option == "Show Chart":
    st.write("chart would appear here")
else:
    st.write("table would appear here ")

with st.form("login_form"):
    username=st.text_input("username")
    password=st.text_input("password",type="password")
    submitted = st.form_submit_button("login")

st.image("https://www.vecteezy.com/photo/53733179-every-detail-of-a-sleek-modern-car-captured-in-close-up",caption="sample Image ")
st.video("https://www.youtube.com/watch?v=rTkgv_mtCeo")
