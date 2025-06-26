import streamlit as st
import os
import tempfile
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()

        all_docs = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Load the document using the temporary file path
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs) #splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector OpenAI embeddings

st.title("RAG app using Nvidia NIM")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

# File uploader widget for user to upload PDFs
uploaded_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)

if st.button("Process Documents"):
    if uploaded_files:
        start_time = time.time()
        vector_embedding(uploaded_files)
        processing_time = time.time() - start_time
        st.write(f"Vector Store DB Is Ready. Processing Time: {processing_time:.2f} seconds")
    else:
        st.write("Please upload at least one PDF document.")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        query_start_time = time.time()
        response = retrieval_chain.invoke({'input': prompt1})
        query_time = time.time() - query_start_time
        st.write(f"Response Time: {query_time:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please upload and process documents first.")

