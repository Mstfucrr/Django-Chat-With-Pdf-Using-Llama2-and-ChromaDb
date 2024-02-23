# Llama API

Welcome to the Llama API, a Django-based web service that leverages language models and document retrieval for answering user queries.

## Getting Started

### Prerequisites
- [Python](https://www.python.org/downloads/) (version 3.11.0)
- [Django](https://www.djangoproject.com/) (version 5.0.2)
- [torch](https://pytorch.org/get-started/locally/) (version 2.0.1)
- [transformers](https://huggingface.co/transformers/installation.html) (version 4.33.0)
- [sentence-transformers](https://www.sbert.net) (version 2.2.2)
- [langchain](https://python.langchain.com/docs/get_started/introduction) (version 0.0.300)
- [chromadb](https://www.trychroma.com) (version 0.4.12)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Mstfucrr/Django-Chat-With-Pdf-Using-Llama2-and-ChromaDb.git
cd llama_django
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the Django app:

```bash
python manage.py migrate
python manage.py runserver
```

The Llama API should now be accessible at `http://localhost:8000/`.

## Usage

### API Endpoints

- **Welcome Message**

  GET `/api/`

  ```json
  {"message": "Welcome to the Llama API"}
  ```

- **Query the Model**

  POST `/api/query`

  **Request Body:**
  ```json
  {"query": "What is SMO"}
  ```

  **Response:**
  ```json
  {
    "result": " SMO stands for Simulated Annealing with Mutation Operator.",
    "status": "success",
    "message": "Query processed successfully"
  }
  ```

- **Get Available Docs**

  GET `/api/get_docs`

  **Response:**
  ```json
   {
    "message": "Success",
    "data": [
      {
        "source": "Document Source 1",
        "content": "Page content of Document 1"
      },
      {
        "source": "Document Source 2",
        "content": "Page content of Document 2"
      },
    ]
  }
  ```

## Components

- **Language Model (LLM)**
  - Loaded with [Hugging Face Transformers](https://huggingface.co/transformers/)
  - Model ID: `meta-llama/Llama-2-7b-chat-hf`

- **PDF Loader**
  - Loads documents from a specified directory using [langchain](https://github.com/langchain-ai/langchain)

- **Text Chunk Splitter**
  - Splits documents into chunks for efficient processing

- **Embeddings**
  - Utilizes [sentence-transformers](https://www.sbert.net/) for generating text embeddings

- **Vector Database**
  - Chroma Vector Store from [langchain](https://github.com/langchain-ai/langchain) to store document vectors

- **Retrieval QA**
  - Uses a combination of LLM, vector database, and retrieval chain for answering queries

## Acknowledgments

- [langchain](https://github.com/langchain-ai/langchain)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---
