# 🧠 AI Research Navigator

**AI Research Navigator** is a semantic search and question-answering system designed for exploring papers published on [arXiv.org](https://arxiv.org). It leverages modern NLP tools such as **LangChain**, **HuggingFace Transformers**, and **Vector Databases** to help researchers build a personalized knowledge base and query it intelligently using LLMs.

---

## ✨ Features

- 🔍 Search and retrieve papers from arXiv using keyword-based API queries  
- 🗂️ Build a local knowledge base with metadata (title, abstract, authors, etc.)  
- 🧠 Perform semantic search over the paper corpus using vector embeddings  
- 🤖 Ask questions to a large language model based on the ingested papers  
- 🧪 Support for OpenAI, Gemini, and HuggingFace LLMs  

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/DostiAziz/Archive_reserach_navigator.git
cd Archive_reserach_navigator
```
### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Alternatively, you can use `make` to install all dependencies:
```bash
make install
```

### 4. Set up environment variables
Create a `.env` file in the root directory and add your API keys
```env
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_api_key
GEMINI_API_KEY=your_google_api_key
```

# Docker commands

## Build the image
```bash
docker build -t ai-research-navigator .


docker run -p 8501:8501 --env-file .env ai-research-navigator
```


## Using Docker Compose
```bash
docker compose build .

docker compose up
```

##  Run application with streamlit
```bash
streamlit run Main.py
```

### Project Structure
```
.
Archive_reserach_navigator/
├── 📁 src/                       # Source code directory
│   ├── Main.py                # Main Streamlit app logic
│   ├── 📁 pages/                 # Streamlit pages
│   ├── 📁 models/                # Data processing and LLM interactions
│   └── 📁 utils/                 # Utility functions and helpers
├── 📁 data/                      # Vector database storage
├── 📁 logs/                      # Application logs
├── 🐳 docker-compose.yml         # Docker Compose configuration
├── 🐳 Dockerfile                 # Docker image definition
├── 📋 requirements.txt           # Python dependencies
├── ⚙️ .env.example              # Environment variables template
├── 🛠️ Makefile                  # Build automation
└── 📖 README.md                  # This file
```

## ✅ Example Workflow
Search papers using keyword input in the app.

Select papers to include in your local vector store.

Ask natural language questions such as:

"What are the main findings in transformer-based speech models?"

"Summarize the latest GAN papers on medical imaging."

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Dosti Aziz - [@Dosti](https://github.com/DostiAziz)