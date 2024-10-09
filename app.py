import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx

import subprocess
import os
import json
import zipfile
import io
from typing import List, Dict
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
from langchain.chains import RetrievalQA

from langchain.chains import ConversationChain

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv

OLLAMA_BASE_URL = "http://localhost:11434"

def run_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def get_available_models() -> List[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return [line.split()[0] for line in result.stdout.split('\n')[1:] if line]
    except FileNotFoundError:
        st.error("The 'ollama' tool is not installed or not found in the environment.")
        return []
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return []

def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    return re.sub(r'[^\w\-_\.]', '_', filename.replace(' ', '_'))

def create_or_update_file(project_dir: str, file_content: str, file_path: str) -> str:
    try:
        sanitized_path = os.path.normpath(sanitize_filename(file_path))
        full_path = os.path.join(project_dir, sanitized_path)
        
        if not full_path.startswith(os.path.normpath(project_dir)):
            raise ValueError("File path cannot be outside the project directory")
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        return f"Successfully created/updated file: {sanitized_path}"
    except Exception as e:
        return f"Error creating/updating file {sanitized_path}: {str(e)}"


def load_document(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_file_path) if file_extension == '.pdf' else TextLoader(temp_file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []
    finally:
        os.unlink(temp_file_path)


def process_document(documents):
    texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    try:
        return FAISS.from_documents(texts, HuggingFaceEmbeddings())
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def init_session_state():
    default_values = {
        'project_info': {},
        'project_started': False,
        'project_state': "Project not started. Waiting for project description...",
        'project_dir': None,
        'vectorstore': None,
        'messages': [],
        'mode': "README Generator",
        'git_initialized': False,
        'current_file': None,
        'web_scraping_results': None,
    }
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def save_project_state():
    if st.session_state.project_dir:
        state_file = os.path.join(st.session_state.project_dir, "project_state.json")
        with open(state_file, "w") as f:
            json.dump({
                "project_info": st.session_state.project_info,
                "project_state": st.session_state.project_state,
                "messages": st.session_state.messages,
                "git_initialized": st.session_state.git_initialized
            }, f)

def load_project_state():
    if st.session_state.project_dir:
        state_file = os.path.join(st.session_state.project_dir, "project_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                st.session_state.project_info = state["project_info"]
                st.session_state.project_state = state["project_state"]
                st.session_state.messages = state["messages"]
                st.session_state.git_initialized = state.get("git_initialized", False)

def set_page_config():
    st.set_page_config(page_title="Dev'sUI", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #F7F7F7; 
    }
    .stSidebar {
        background-color: #000000; 
        color: #F7F7F7;
    }
    .stButton>button {
        background-color: #93DEFF; 
        color: #000000; 
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        background-color: #F7F7F7; 
        color: #000000; 
    }
    .stHeader {
        background-color: #000000; 
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #F7F7F7;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_content():
    with st.sidebar:
        st.title("Dev'sUI")
        selected_mode = option_menu(
            menu_title=None,
            options=["README Generator", "Document Chat", "General Chat", "Web Scraping"],
            icons=['file-text', 'chat-dots', 'chat', 'globe'],
            default_index=0,
            
        )
        st.session_state.mode = selected_mode

        available_models = get_available_models()
        if available_models:
            st.session_state.selected_model = st.selectbox("Choose a model", available_models)
        else:
            st.session_state.selected_model = None

        if st.session_state.project_dir:
            if st.button("Export Project", key="export_project"):
                export_project()

    return st.session_state.selected_model

def export_project():
    if st.session_state.project_dir:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(st.session_state.project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, st.session_state.project_dir)
                    zip_file.write(file_path, arc_name)
        
        zip_buffer.seek(0)
        st.sidebar.download_button(
            label="Download Project ZIP",
            data=zip_buffer,
            file_name=f"{os.path.basename(st.session_state.project_dir)}.zip",
            mime="application/zip"
        )

def readme_generator_mode(selected_model: str):
    load_project_state()
    project_dir = st.text_input("Enter the local path of the project directory:")
    if st.button("Generate README") and project_dir:
        with st.spinner("Generating README..."):
            readme_content = generate_readme(project_dir, selected_model)
            st.session_state.current_file = "README.md"
            st.session_state.project_dir = project_dir
            update_message = create_or_update_file(project_dir, readme_content, "README.md")
            if "Error" in update_message:
                st.error(update_message)
            else:
                st.success(update_message)
            save_project_state()
            
    
    if st.session_state.current_file:
        st.write(f"Editing: {st.session_state.current_file}")
        file_path = os.path.join(st.session_state.project_dir, st.session_state.current_file)
        with open(file_path, "r") as f:
            file_content = f.read()
        edited_content = st.text_area("Edit file content:", value=file_content, height=300)
        if st.button("Save Changes"):
            update_message = create_or_update_file(st.session_state.project_dir, edited_content, st.session_state.current_file)
            if "Error" in update_message:
                st.error(update_message)
            else:
                st.success(update_message)
            save_project_state()
    
    if st.button("Delete Project State"):
        if st.session_state.project_dir:
            state_file = os.path.join(st.session_state.project_dir, "project_state.json")
            if os.path.exists(state_file):
                os.remove(state_file)
                st.success("Project state deleted successfully.")
            else:
                st.error("No project state file found to delete.")
        else:
            st.error("No project directory set.")

def generate_readme(project_dir: str, selected_model: str) -> str:
    readme_path = os.path.join(project_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            existing_readme = f.read()
    else:
        existing_readme = ""

    llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
    template = """You are an AI developer assistant tasked with generating a README file for a project. The README should include an overview, installation instructions, usage examples, and any other relevant information.

    Project Directory:
    {project_dir}

    Existing README Content:
    {existing_readme}

    AI Developer:"""
    conversation = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["project_dir", "existing_readme"], template=template),
        verbose=True,
        memory=ConversationBufferMemory(input_key="project_dir", memory_key="chat_history"),
    )

    response = conversation.predict(project_dir=project_dir, existing_readme=existing_readme)
    return response

def document_chat_mode(selected_model: str):
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "current_document" not in st.session_state:
        st.session_state.current_document = None

    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #F7F7F7;
    }
    .stButton>button {
        background-color: #93DEFF;
        color: #000000;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #000000;
        color: #F7F7F7;
        border-color: #93DEFF;
    }
    .stTabs>div>div>div {
        background-color: #000000;
        color: #F7F7F7;
    }
    .stTabs>div>div>div[data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs>div>div>div[data-baseweb="tab"] {
        background-color: #000000;
        color: #F7F7F7;
        border-radius: 5px 5px 0 0;
    }
    .stTabs>div>div>div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #93DEFF;
        color: #000000;
    }
    .stMarkdown {
        color: #F7F7F7;
    }
    .stExpander {
        background-color: #000000;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Document Chat")
    
    tab1, tab2 = st.tabs(["📁 Manage Documents", "💬 Chat"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or MD)", type=["pdf", "txt", "md"])
            if uploaded_file is not None:
                if uploaded_file.name not in st.session_state.documents:
                    if st.button("Process Document"):
                        with st.spinner("Processing document..."):
                            documents = load_document(uploaded_file)
                            vectorstore = process_document(documents)
                            if vectorstore:
                                st.session_state.documents[uploaded_file.name] = {
                                    "vectorstore": vectorstore,
                                    "content": documents
                                }
                                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                else:
                    st.info(f"Document '{uploaded_file.name}' already processed.")

        with col2:
            st.subheader("Manage Documents")
            for doc_name in st.session_state.documents.keys():
                with st.expander(doc_name):
                    col1, col2 = st.columns(2)
                    if col1.button("Select", key=f"select_{doc_name}"):
                        st.session_state.current_document = doc_name
                    if col2.button("Delete", key=f"delete_{doc_name}"):
                        del st.session_state.documents[doc_name]
                        if st.session_state.current_document == doc_name:
                            st.session_state.current_document = None
                        st.experimental_rerun()

    with tab2:
        if st.session_state.current_document:
            st.info(f"Current document: {st.session_state.current_document}")
            
            llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
            
            doc_chat_input = st.text_input("Ask a question about the document:")
            if doc_chat_input:
                with st.spinner("Searching for answer..."):
                    vectorstore = st.session_state.documents[st.session_state.current_document]["vectorstore"]
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm, 
                        chain_type="stuff", 
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                    )
                    answer = qa_chain.run(doc_chat_input)
                    
                    st.markdown("### Answer:")
                    st.markdown(f'<div style="background-color: #000000; color: #F7F7F7; padding: 10px; border-radius: 5px; border: 1px solid #93DEFF;">{answer}</div>', unsafe_allow_html=True)
                    
                    with st.expander("View Relevant Passages"):
                        relevant_docs = vectorstore.similarity_search(doc_chat_input, k=3)
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Passage {i}:**")
                            st.markdown(f'<div style="background-color: #000000; color: #F7F7F7; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #93DEFF;">{doc.page_content}</div>', unsafe_allow_html=True)
            
            if st.button("View Document Content"):
                with st.expander("Document Content", expanded=True):
                    content = st.session_state.documents[st.session_state.current_document]["content"]
                    for i, page in enumerate(content, 1):
                        st.markdown(f"**Page {i}:**")
                        st.markdown(f'<div style="background-color: #000000; color: #F7F7F7; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #93DEFF;">{page.page_content}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please select a document to start chatting.")


def general_chat_mode(selected_model: str):
    if "general_chat_messages" not in st.session_state:
        st.session_state.general_chat_messages = []
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory()

    llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
    conversation = ConversationChain(llm=llm, memory=st.session_state.chat_memory)

    for message in st.session_state.general_chat_messages:
        st.chat_message(message["role"]).write(message["content"])

    user_input = st.chat_input("Chat with the AI:", key="general_chat_input")

    if user_input:
        st.session_state.general_chat_messages.append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)

        with st.chat_message("ai"):
            with st.spinner("AI is thinking..."):
                try:
                    response = conversation.predict(input=user_input)
                    st.write(response)
                    st.session_state.general_chat_messages.append({"role": "ai", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    
    if st.button("Clear Chat History"):
        st.session_state.general_chat_messages = []
        st.session_state.chat_memory.clear()
        

def scrape_data(soup, data_to_scrape):
    scraped_data = {}
    if data_to_scrape["title"]:
        scraped_data["title"] = soup.title.string if soup.title else "No title found"
    if data_to_scrape["meta"]:
        scraped_data["meta"] = {meta['name']: meta['content'] for meta in soup.find_all('meta', attrs={'name': True, 'content': True})}
    if data_to_scrape["headers"]:
        scraped_data["headers"] = {f"h{i}": [h.text for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
    if data_to_scrape["paragraphs"]:
        scraped_data["paragraphs"] = [p.text for p in soup.find_all('p')]
    if data_to_scrape["links"]:
        scraped_data["links"] = [{'text': a.text, 'href': a['href']} for a in soup.find_all('a', href=True)]
    if data_to_scrape["images"]:
        scraped_data["images"] = [{'src': img['src'], 'alt': img.get('alt', '')} for img in soup.find_all('img', src=True)]
    if data_to_scrape["tables"]:
        scraped_data["tables"] = [pd.read_html(str(table))[0].to_dict() for table in soup.find_all('table')]
    if data_to_scrape["lists"]:
        scraped_data["lists"] = {
            'ul': [{'items': [li.text for li in ul.find_all('li')]} for ul in soup.find_all('ul')],
            'ol': [{'items': [li.text for li in ol.find_all('li')]} for ol in soup.find_all('ol')]
        }
    if data_to_scrape["scripts"]:
        scraped_data["scripts"] = [script.string for script in soup.find_all('script') if script.string]
    if data_to_scrape["styles"]:
        scraped_data["styles"] = [style.string for style in soup.find_all('style') if style.string]
    return scraped_data

def web_scraping_mode(selected_model: str):
    st.write("### Web Scraping")
    url = st.text_input("Enter a URL to scrape:")
    st.write("Select the data you want to scrape:")
    columns = st.columns(10)
    data_to_scrape = {
        "title": columns[0].checkbox("Title", value=True),
        "meta": columns[1].checkbox("Meta Information", value=True),
        "headers": columns[2].checkbox("Headers", value=True),
        "paragraphs": columns[3].checkbox("Paragraphs", value=True),
        "links": columns[4].checkbox("Links", value=True),
        "images": columns[5].checkbox("Images", value=True),
        "tables": columns[6].checkbox("Tables", value=True),
        "lists": columns[7].checkbox("Lists", value=True),
        "scripts": columns[8].checkbox("Scripts", value=True),
        "styles": columns[9].checkbox("Styles", value=True)
    }

    if st.button("Scrape") and url:
        with st.spinner("Scraping website..."):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                st.session_state.scraped_data = scrape_data(soup, data_to_scrape)
                st.success("Scraping completed successfully!")
            except Exception as e:
                st.error(f"Error scraping website: {str(e)}")

    if "scraped_data" in st.session_state:
        st.write("#### Scraping Results")
        for key, value in st.session_state.scraped_data.items():
            st.subheader(key.capitalize())
            if isinstance(value, str):
                st.write(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    st.write(f"- {sub_key}: {sub_value}")
            elif isinstance(value, list):
                if key == "tables":
                    for i, table in enumerate(value):
                        st.write(f"Table {i+1}")
                        st.dataframe(pd.DataFrame(table))
                else:
                    for item in value[:5]:
                        st.write(f"- {item}")
                    if len(value) > 5:
                        st.write(f"... and {len(value) - 5} more items")

        st.subheader("Ask a question about the scraped data")
        query = st.text_input("Enter your query:")
        if st.button("Get Answer"):
            answer = answer_query_with_model(query, st.session_state.scraped_data, selected_model)
            st.write("Answer:", answer)

def answer_query_with_model(query: str, data: dict, selected_model: str) -> str:
    
    data_str = json.dumps(data, indent=2)

    
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])

    prompt = f"""
    You are an AI assistant. Here is the scraped data from a website:
    {data_str}

    Based on this data, answer the following question:
    {query}
    """

    response = llm(prompt)
    return response

def main():
    set_page_config()
    init_session_state()
    selected_model = sidebar_content()
    
    if st.session_state.mode == "README Generator":
        readme_generator_mode(selected_model)
    elif st.session_state.mode == "Document Chat":
        document_chat_mode(selected_model)
    elif st.session_state.mode == "General Chat":
        general_chat_mode(selected_model)
    elif st.session_state.mode == "Web Scraping":
        web_scraping_mode(selected_model)

def delete_history():
    if st.session_state.project_dir:
        state_file = os.path.join(st.session_state.project_dir, "project_state.json")
        if os.path.exists(state_file):
            os.remove(state_file)
            st.success("Project state deleted successfully.")
        else:
            st.error("No project state file found to delete.")

if __name__ == "__main__":
    main()
    delete_history()