
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Streamlit page
st.set_page_config(page_title="RAG Tutorial", page_icon="📚")

st.title("RAG Application using LangChain")
st.write("Ask questions about your C++ notes.")


# Load document and create FAISS database
@st.cache_resource
def create_vector_database():

    loader = TextLoader("CPP.txt", encoding="utf-8")
    text_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    final_documents = text_splitter.split_documents(
        text_documents
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = FAISS.from_documents(
        final_documents,
        embeddings
    )

    return db


# LLM
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question "
        "using the provided context. If the answer is not "
        "present in the context, say you could not find it "
        "in the document."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])


# User input
question = st.text_input(
    "Enter your question about C++:"
)


if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching document and generating answer..."):

                # Create or load the vector database
                db = create_vector_database()

                # Retrieve relevant document chunks
                retriever = db.as_retriever(
                    search_kwargs={"k": 3}
                )

                relevant_docs = retriever.invoke(question)

                # Combine retrieved text
                context = "\n\n".join(
                    doc.page_content
                    for doc in relevant_docs
                )

                # Generate answer
                chain = prompt | llm | StrOutputParser()

                answer = chain.invoke({
                    "context": context,
                    "question": question
                })

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View retrieved document content"):
                st.write(context)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info(
                "Check that Ollama is running and the "
                "required models and Python packages are installed."
            )
