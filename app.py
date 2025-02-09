import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage, AIMessage

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit UI
st.set_page_config(page_title="Legal Document AI", layout="wide")
st.title("📜 Legal Document AI Assistant")

# Get OpenAI API key from user
api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# File uploader
uploaded_files = st.file_uploader("📂 Upload legal documents (PDF/DOCX)", type=["pdf", "docx"],
                                  accept_multiple_files=True)

if api_key and uploaded_files:
    os.environ["OPENAI_API_KEY"] = api_key  # Set API Key
    docs = []

    # Load Documents
    for uploaded_file in uploaded_files:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue

        docs.extend(loader.load())

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    document_chunks = text_splitter.split_documents(docs)

    # Store in Vector Database
    vector_db = Chroma.from_documents(document_chunks, OpenAIEmbeddings())


    # Define Retrieval Chain
    def _get_context_retriever_chain(vector_db, llm):
        retriever = vector_db.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}")
        ])
        return create_history_aware_retriever(llm, retriever, prompt=prompt
                                              )


    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


    def get_legal_summary_chain(llm):
        retriever_chain = _get_context_retriever_chain(vector_db, llm)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are an AI assistant summarizing legal documents.
             Extract details such as case title, court name, judgment date, parties involved, 
             legal provisions, arguments, and final verdict.
             Context: {context}"""),
            ("user", "{messages}")
        ])

        summarization_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, summarization_chain)


    conversation_rag_chain = get_legal_summary_chain(llm)

    # User Input
    query = st.text_input("📝 Enter case name or query:")

    if query:
        messages = query
        response_message = ""
        with st.spinner("⏳ Generating summary..."):
            for chunk in conversation_rag_chain.pick("answer").stream({"input": messages, "messages": []}):
                response_message += chunk

        st.subheader("📄 Legal Summary:")
        st.write(response_message)
