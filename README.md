# AI Developer Assistant

Welcome to the AI Developer Assistant! This tool is designed to help you build projects, chat with documents, have general conversations with AI models, and perform web scraping tasks. The assistant leverages advanced AI technologies to streamline your development workflow, providing a versatile platform for a range of tasks.

## Features

- **Project Building Mode:**
  - Guides you through the process of setting up and managing a project.
  - Assists in creating, updating, and viewing project files.
  - Integrates with Git for version control, including initializing repositories, committing changes, and managing branches.
  - Uses the Ollama model for generating code and project structure suggestions.

- **Document Chat Mode:**
  - Allows you to upload and chat with documents (PDF, TXT).
  - Utilizes advanced vector storage (FAISS) and retrieval to answer questions based on the content of uploaded documents.

- **General Chat Mode:**
  - Provides an interface to have general conversations with AI models.
  - Uses the Ollama model to generate responses and engage in meaningful discussions.

- **Web Scraping Mode:**
  - Enables scraping of web content including text, links, images, tables, lists, scripts, and styles.
  - Provides options to export scraped data in JSON, CSV, or HTML formats.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Ollama AI models installed and running locally

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AKN414-IND/ai-developer-assistant.git
   cd ai-developer-assistant
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install and run the Ollama model locally:
   ```bash
   curl https://ollama.ai/install.sh | sh
   ollama run <model_name>
   ```
   Ensure the Ollama service is running on http://localhost:11434.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Project Building:**
   - Navigate to the "Project Building" mode.
   - Create a new project or select an existing one.
   - Use the file tree view to navigate and edit project files.
   - Utilize Git integration for version control.

2. **Document Chat:**
   - Upload a document (PDF or TXT) and interact with it by asking questions.
   - The AI will retrieve relevant information from the document and provide answers.

3. **General Chat:**
   - Use the general chat mode for any queries or conversations with the AI.

4. **Web Scraping:**
   - Enter a URL and select the types of data you want to scrape.
   - View the scraped data and export it in your preferred format.

## Customization

You can customize the AI model, project settings, and features through the Streamlit sidebar. Select different modes, choose a model, and monitor the project state as you work.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. Ensure your code follows the existing style and passes all tests.

## License

This project is licensed under the MIT License. See the [LICENSE](/mit.md) file for more details.

## Creator

This project was created by [AKN414-IND](https://github.com/AKN414-IND).
