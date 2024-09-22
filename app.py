import streamlit as st
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

import errno
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, TextLexer
from pygments.formatters.html import HtmlFormatter
from langchain.chains import RetrievalQA
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
        'mode': "Project Building",
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

def sidebar_content():
    st.sidebar.title("AI Developer Assistant")
    st.session_state.mode = st.sidebar.radio("Choose a mode", ["Project Building", "Document Chat", "General Chat", "Web Scraping"])
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Choose a model", available_models) if available_models else None
    if "project_state" in st.session_state:
        st.sidebar.text_area("Current State", st.session_state.project_state, height=100)
    if st.session_state.project_dir:
        if st.sidebar.button("Export Project"):
            export_project()
    return selected_model

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

def project_building_mode(selected_model: str):
    if not st.session_state.project_started:
        st.write("Please provide a brief description of your project and the desired project directory name.")
        project_description = st.text_area("Project Description:", height=150)
        project_name = st.text_input("Project Directory Name:")
        if st.button("Start Project") and project_description and project_name:
            sanitized_project_name = sanitize_filename(project_name)
            st.session_state.project_dir = os.path.join(os.getcwd(), sanitized_project_name)
            os.makedirs(st.session_state.project_dir, exist_ok=True)
            st.session_state.project_started = True
            st.session_state.project_info['description'] = project_description
            st.session_state.project_state = f"Project initialized. Directory: {st.session_state.project_dir}"
            
            git_init_message = init_git_repo(st.session_state.project_dir)
            st.session_state.git_initialized = "successfully" in git_init_message.lower()
            st.session_state.project_state += f"\n{git_init_message}"
            
            save_project_state()
            st.experimental_rerun()
    elif selected_model and st.session_state.project_started:
        load_project_state()
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
        template = """You are an AI developer assistant tasked with building a base application for a team to work on. Your goal is to create a starting point with dummy data and basic functionality based on the project description provided. Always provide relative file paths when creating or updating files.

        Project Description:
        {project_info}

        Current project state:
        {project_state}

        Chat history:
        {chat_history}

        Human: {human_input}
        AI Developer:"""
        conversation = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["project_info", "project_state", "chat_history", "human_input"], template=template),
            verbose=True,
            memory=ConversationBufferMemory(input_key="human_input", memory_key="chat_history"),
        )

        st.write("### Chat History")
        for message in st.session_state.messages:
            st.write(f"**{message['role']}:** {message['content']}")

        user_input = st.text_input("What would you like the AI developer to do?", key="user_input")
        if st.button("Send") and user_input:
            st.session_state.messages.append({"role": "Human", "content": user_input})
            with st.spinner("AI developer is working..."):
                response = conversation.predict(human_input=user_input, project_info=json.dumps(st.session_state.project_info), project_state=st.session_state.project_state)
            st.session_state.messages.append({"role": "AI Developer", "content": response})
            if "```" in response:
                for i in range(1, len(response.split("```")), 2):
                    file_content = response.split("```")[i].strip()
                    file_path = file_content.split("\n")[0].strip()
                    code = "\n".join(file_content.split("\n")[1:])
                    update_message = create_or_update_file(st.session_state.project_dir, code, file_path)
                    st.session_state.project_state += f"\n{update_message}"
                    if "Error" in update_message:
                        st.error(update_message)
                    else:
                        st.success(update_message)
            save_project_state()
            st.experimental_rerun()

        if st.session_state.git_initialized:
            st.write("### Git Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                commit_message = st.text_input("Commit message:")
                if st.button("Commit Changes") and commit_message:
                    commit_result = git_add_commit(st.session_state.project_dir, commit_message)
                    st.write(commit_result)
            
            with col2:
                new_branch_name = st.text_input("New branch name:")
                if st.button("Create Branch") and new_branch_name:
                    branch_result = git_create_branch(st.session_state.project_dir, new_branch_name)
                    st.write(branch_result)
            
            branches = git_list_branches(st.session_state.project_dir)
            if isinstance(branches, list):
                selected_branch = st.selectbox("Select branch:", branches)
                if st.button("Switch Branch"):
                    switch_result = git_switch_branch(st.session_state.project_dir, selected_branch)
                    st.write(switch_result)
            
            current_branch = git_get_current_branch(st.session_state.project_dir)
            st.write(f"Current branch: {current_branch}")
            
            st.write("### Commit History")
            commit_history = git_get_commit_history(st.session_state.project_dir)
            if isinstance(commit_history, list):
                for commit in commit_history:
                    st.write(f"{commit['hash']} - {commit['message']} (by {commit['author']} on {commit['date']})")
        
        st.write("### Project Files")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("File Explorer")
            project_tree_view()
        
        with col2:
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
            else:
                st.write("Select a file from the File Explorer to edit")

def document_chat_mode(selected_model: str):
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or MD)", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            documents = load_document(uploaded_file)
            vectorstore = process_document(documents)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success("Document processed successfully!")
    if st.session_state.vectorstore:
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
        doc_chat_input = st.text_input("Ask a question about the document:")
        if doc_chat_input:
            with st.spinner("Searching for answer..."):
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vectorstore.as_retriever())
                st.write("Answer:", qa_chain.run(doc_chat_input))

def general_chat_mode(selected_model: str):
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=selected_model, callbacks=[StreamingStdOutCallbackHandler()])
    if "general_chat_messages" not in st.session_state:
        st.session_state.general_chat_messages = []
    for message in st.session_state.general_chat_messages:
        st.write(f"**{message['role']}:** {message['content']}")
    user_input = st.text_input("Chat with the AI:", key="general_chat_input")
    if st.button("Send", key="general_chat_send") and user_input:
        st.session_state.general_chat_messages.append({"role": "Human", "content": user_input})
        with st.spinner("AI is thinking..."):
            response = llm(user_input)
        st.session_state.general_chat_messages.append({"role": "AI", "content": response})
        st.experimental_rerun()

def web_scraping_mode():
    st.write("### Web Scraping")
    url = st.text_input("Enter a URL to scrape:")
    
    st.write("Select the data you want to scrape:")
    data_to_scrape = {
        "title": st.checkbox("Title", value=True),
        "meta": st.checkbox("Meta Information", value=True),
        "headers": st.checkbox("Headers", value=True),
        "paragraphs": st.checkbox("Paragraphs", value=True),
        "links": st.checkbox("Links", value=True),
        "images": st.checkbox("Images", value=True),
        "tables": st.checkbox("Tables", value=True),
        "lists": st.checkbox("Lists", value=True),
        "scripts": st.checkbox("Scripts", value=True),
        "styles": st.checkbox("Styles", value=True)
    }
    
    if st.button("Scrape") and url:
        with st.spinner("Scraping website..."):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                st.session_state.web_scraping_results = {}
                
                if data_to_scrape["title"]:
                    st.session_state.web_scraping_results["title"] = soup.title.string if soup.title else "No title found"
                if data_to_scrape["meta"]:
                    st.session_state.web_scraping_results["meta"] = {meta['name']: meta['content'] for meta in soup.find_all('meta', attrs={'name': True, 'content': True})}
                if data_to_scrape["headers"]:
                    st.session_state.web_scraping_results["headers"] = {f"h{i}": [h.text for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
                if data_to_scrape["paragraphs"]:
                    st.session_state.web_scraping_results["paragraphs"] = [p.text for p in soup.find_all('p')]
                if data_to_scrape["links"]:
                    st.session_state.web_scraping_results["links"] = [{'text': a.text, 'href': a['href']} for a in soup.find_all('a', href=True)]
                if data_to_scrape["images"]:
                    st.session_state.web_scraping_results["images"] = [{'src': img['src'], 'alt': img.get('alt', '')} for img in soup.find_all('img', src=True)]
                if data_to_scrape["tables"]:
                    st.session_state.web_scraping_results["tables"] = [pd.read_html(str(table))[0].to_dict() for table in soup.find_all('table')]
                if data_to_scrape["lists"]:
                    st.session_state.web_scraping_results["lists"] = {
                        'ul': [{'items': [li.text for li in ul.find_all('li')]} for ul in soup.find_all('ul')],
                        'ol': [{'items': [li.text for li in ol.find_all('li')]} for ol in soup.find_all('ol')]
                    }
                if data_to_scrape["scripts"]:
                    st.session_state.web_scraping_results["scripts"] = [script.string for script in soup.find_all('script') if script.string]
                if data_to_scrape["styles"]:
                    st.session_state.web_scraping_results["styles"] = [style.string for style in soup.find_all('style') if style.string]
                
                st.success("Scraping completed successfully!")
            except Exception as e:
                st.error(f"Error scraping website: {str(e)}")
    
    if st.session_state.web_scraping_results:
        st.write("#### Scraping Results")
        
        for key, value in st.session_state.web_scraping_results.items():
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
        
        st.subheader("Export Options")
        export_format = st.selectbox("Choose export format", ["JSON", "CSV", "HTML"])
        if st.button("Export Data"):
            if export_format == "JSON":
                json_str = json.dumps(st.session_state.web_scraping_results, indent=2)
                st.download_button(label="Download JSON", data=json_str, file_name="scraping_results.json", mime="application/json")
            elif export_format == "CSV":
                csv_data = io.StringIO()
                writer = csv.writer(csv_data)
                for key, value in st.session_state.web_scraping_results.items():
                    if isinstance(value, list):
                        writer.writerow([key] + value)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow([f"{key}_{sub_key}", sub_value])
                    else:
                        writer.writerow([key, value])
                st.download_button(label="Download CSV", data=csv_data.getvalue(), file_name="scraping_results.csv", mime="text/csv")
            elif export_format == "HTML":
                html_str = "<html><body>"
                for key, value in st.session_state.web_scraping_results.items():
                    html_str += f"<h2>{key}</h2>"
                    if isinstance(value, str):
                        html_str += f"<p>{value}</p>"
                    elif isinstance(value, dict):
                        html_str += "<ul>"
                        for sub_key, sub_value in value.items():
                            html_str += f"<li><strong>{sub_key}:</strong> {sub_value}</li>"
                        html_str += "</ul>"
                    elif isinstance(value, list):
                        html_str += "<ul>"
                        for item in value:
                            html_str += f"<li>{item}</li>"
                        html_str += "</ul>"
                html_str += "</body></html>"
                st.download_button(label="Download HTML", data=html_str, file_name="scraping_results.html", mime="text/html")

def main():
    st.set_page_config(page_title="AI Developer Assistant", layout="wide")
    init_session_state()
    selected_model = sidebar_content()
    
    if st.session_state.mode == "Project Building":
        project_building_mode(selected_model)
    elif st.session_state.mode == "Document Chat":
        document_chat_mode(selected_model)
    elif st.session_state.mode == "General Chat":
        general_chat_mode(selected_model)
    elif st.session_state.mode == "Web Scraping":
        web_scraping_mode()

def project_tree_view():
    if st.session_state.project_dir:
        for root, dirs, files in os.walk(st.session_state.project_dir):
            level = root.replace(st.session_state.project_dir, '').count(os.sep)
            indent = '&nbsp;&nbsp;&nbsp;&nbsp;' * level
            folder_name = os.path.basename(root)
            st.markdown(f"{indent}📁 {folder_name}", unsafe_allow_html=True)
            subindent = '&nbsp;&nbsp;&nbsp;&nbsp;' * (level + 1)
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), st.session_state.project_dir)
                if st.button(f"{subindent}📄 {file}", key=file_path):
                    st.session_state.current_file = file_path

def init_git_repo(project_dir: str) -> str:
    try:
        subprocess.run(["git", "init"], cwd=project_dir, check=True, capture_output=True, text=True)
        return "Git repository initialized successfully."
    except subprocess.CalledProcessError as e:
        return f"Error initializing Git repository: {e.stderr}"

def git_add_commit(project_dir: str, commit_message: str) -> str:
    try:
        subprocess.run(["git", "add", "."], cwd=project_dir, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=project_dir, check=True, capture_output=True, text=True)
        return "Changes committed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error committing changes: {e.stderr}"

def git_create_branch(project_dir: str, branch_name: str) -> str:
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=project_dir, check=True, capture_output=True, text=True)
        return f"Branch '{branch_name}' created and switched to successfully."
    except subprocess.CalledProcessError as e:
        return f"Error creating branch: {e.stderr}"

def git_switch_branch(project_dir: str, branch_name: str) -> str:
    try:
        subprocess.run(["git", "checkout", branch_name], cwd=project_dir, check=True, capture_output=True, text=True)
        return f"Switched to branch '{branch_name}' successfully."
    except subprocess.CalledProcessError as e:
        return f"Error switching branch: {e.stderr}"

def git_list_branches(project_dir: str) -> List[str]:
    try:
        result = subprocess.run(["git", "branch"], cwd=project_dir, check=True, capture_output=True, text=True)
        return [branch.strip() for branch in result.stdout.split('\n') if branch.strip()]
    except subprocess.CalledProcessError as e:
        st.error(f"Error listing branches: {e.stderr}")
        return []

def git_get_current_branch(project_dir: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_dir, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error getting current branch: {e.stderr}")
        return "Unknown"

def git_get_commit_history(project_dir: str) -> List[Dict[str, str]]:
    try:
        result = subprocess.run(["git", "log", "--pretty=format:%H|%an|%ad|%s"], cwd=project_dir, check=True, capture_output=True, text=True)
        commits = []
        for line in result.stdout.split('\n'):
            if line:
                hash, author, date, message = line.split('|', 3)
                commits.append({
                    "hash": hash[:7],
                    "author": author,
                    "date": date,
                    "message": message
                })
        return commits
    except subprocess.CalledProcessError as e:
        st.error(f"Error getting commit history: {e.stderr}")
        return []

if __name__ == "__main__":
    main()