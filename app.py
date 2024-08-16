import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata else {}

# Initialize the model
LOCAL_MODEL = "llama3.1"
EMBEDDING = "nomic-embed-text"
llm = Ollama(base_url="http://localhost:11434", model=LOCAL_MODEL, verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Streamlit UI
st.title("PDF Content-Based Query Answering System")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Convert uploaded file to a byte stream
    pdf_file = BytesIO(uploaded_file.getvalue())

    # Read PDF content using PyPDF2
    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    all_text = ""
    for page in range(number_of_pages):
        page_text = reader.pages[page].extract_text()
        if page_text:
            all_text += page_text + "\n"
    
    # Display the extracted text
    st.text_area("Extracted Text", all_text, height=300)

    # Text processing setup
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    doc = Document(all_text)
    all_splits = text_splitter.split_documents([doc])

    persist_directory = 'data'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model=EMBEDDING),
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()
    template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )


    # Query handling
    
    query = st.text_input("Enter your question:")
    if query:
        response = qa_chain.invoke({"query": query})
        st.write(response)
