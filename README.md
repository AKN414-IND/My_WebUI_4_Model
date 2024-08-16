markdown
# PDF Content-Based Query Answering System

This application is designed to extract text from uploaded PDF files and answer queries based on the content using a local Ollama LLM model. It uses Streamlit for the web interface, PyPDF2 for PDF text extraction, and LangChain for integrating the LLM with a custom retrieval-based Q&A setup.

## Features

- PDF text extraction
- Content-based query answering
- Streamlit web interface

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AKN414-IND/Chat_pdf_local_clone_llama3.1
   cd your-repository-directory
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Local LLM Model Setup:**
   Ensure that your local LLM model server is set up and running at `http://localhost:11434`. This setup should include the necessary models specified in your script (`llama3.1`).

## Running the Application

To run the application:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your web browser to interact with the application.

## Usage

1. **Upload a PDF**: Use the file uploader to select and upload a PDF file.
2. **Extracted Text**: After uploading, the application will display the extracted text from the PDF in a text area.
3. **Enter a Query**: Input your question in the text input field and submit.
4. **View the Answer**: The application will display the answer based on the content extracted from the PDF.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

For any queries or technical issues, please open an issue on the GitHub repository or contact [Your Email](mailto:arunknair2003@gmail.com).
