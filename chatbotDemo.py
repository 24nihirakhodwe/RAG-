import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#page config
st.set_page_config(page_title="C++ RAG CHATBOT")
st.title("C++ RAG CHATBOT")
st.write("ask any questions related to c++ introduction")

#load environment variables
@st.cache_resource
def load_vectorstore():
    #load document
    loader=TextLoader("CPP.txt",encoding="utf-8")
    text_documents=loader.load()

    #split text
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )   

    final_documents=text_splitter.split_documents(text_documents)

    #embeddings 
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(embeddings)


    db=FAISS.from_documents(final_documents,embeddings)
    return db

db=load_vectorstore()

query = st.text_input ("enter your questions for c++:")
if query:
     docs=db.similarity_search(query,k=3)
     st.subheader("Retrived context:")
     for i ,doc in enumerate(docs):
          st.markdown(f"Result{i+1}:")
          st.write(doc.page_content)
         
