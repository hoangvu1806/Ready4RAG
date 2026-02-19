
<div align="center">

# **Ready4RAG**
### High-Precision Dual-Layer RAG Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](https://qdrant.tech/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange)](https://networkx.org/)

**A next-generation ingestion system converting complex PDFs into value using Vision LLMs and Hybrid Memory (Vector + Graph).**

![RAG Pipeline Architecture](assets/pipeline_architecture.png)

</div>

---


## Features

<table>
  <tr>
    <td><b>Vision-Powered Extraction</b></td>
    <td>Converts PDF to Markdown with near-perfect layout preservation using multimodal LLMs.</td>
  </tr>
  <tr>
    <td><b>Dual-Layer Memory</b></td>
    <td>Stores data in <b>Qdrant</b> (Vectors) for similarity and <b>NetworkX</b> (Graph) for reasoning.</td>
  </tr>
  <tr>
    <td><b>Auto-Graph Construction</b></td>
    <td>Extracts entities (People, Locations, Concepts) and relationships automatically.</td>
  </tr>
  <tr>
    <td><b>Hybrid Chatbot</b></td>
    <td>Interactive chat that retrieves context from both memory layers for grounded answers.</td>
  </tr>
  <tr>
    <td><b>Multi-Provider Support</b></td>
    <td>Plug-and-play with OpenAI, Google Gemini, Groq, or local Ollama models.</td>
  </tr>
</table>

---

## Installation

### Prerequisites
-   Python 3.10+
-   API Key for OpenAI, Gemini, or Groq (or a local Ollama setup)

### Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/hoangvu1806/Ready4RAG.git
    cd Ready4RAG
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment**
    ```bash
    cp .env.example .env
    # Edit .env with your API keys
    ```

---

## Usage

### 1. Extract Content (PDF → Markdown)
Use the `extract.py` wrapper to convert your documents using Vision AI.

```bash
python extract.py data/medical_report.pdf --provider gemini
```
> Output saved to: `output/markdown/medical_report.md`

### 2. Ingest Data (Markdown → Vector/Graph)
Ingest the extracted content into the dual-layer memory.

```bash
python ingest.py ingest output/markdown/medical_report.md
```

### 3. Chat with Data
Start the hybrid chatbot to query your knowledge base.

```bash
python chatbot.py
```
**Pro Tip:** Type `/verbose` in the chat to see exactly what context (text chunks + graph entities) was retrieved!

---


## Architecture

See the architecture diagram in the overview above.

-   **pdf2md**: Handles the visual understanding of documents.
-   **rag_ingest**: Orchestrates chunking, embedding, and graph extraction.
-   **Graph Store**: Directed graph storing entities as nodes and relationships as weighted edges.
-   **Vector Store**: Stores embeddings for chunks, entities, relationships, and summaries.

---

## Configuration

Configure the pipeline via `.env`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PDF2MD_LLM_PROVIDER` | Model for PDF conversion | `gemini` |
| `QDRANT_USE_LOCAL` | Use local file storage vs server | `true` |
| `RAG_ENABLE_ENTITY_EXTRACTION` | Enable/Disable Graph building | `true` |
| `EMBEDDING_PROVIDER` | Embedding model provider | `gemini` |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

<div align="center">

**Built by [Hoang Vu](https://github.com/hoangvu1806)**

</div>
