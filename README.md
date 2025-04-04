# pubmedRAG
Create a customized PubMed vector store and RAG assistant to stay up to date on the latest articles. This RAG system will give you a general summary of a research topic so you can understand the main concepts of any biomedical field. This is perfect for searching that peripheral topics relating to your main research for which you only need a general knowledge.

## How it works

### Web Scraping and Vector Database
Given a user's input research topic, the program uses the PubMed eutils API to search for PubMedCentral IDs and return relavent articles. Then, the LangChain document splitter is called with a fixed chunk size and chunk overlap before embedding the documents into a local FAISS vector storage.

### RAG

I use LangChain and LangGraph to create a retreival augmented generation (RAG) workflow. The workflow consists of one chatbot and 2 states: retrieval and generation. The retrieval state performs a similarity search and retrieves the k-nearest documents to augment the llm query. In the generation state decides, the llm decides whether it has enough information to generate an accurate response. If it does not know the answer, it will simply state that. Furthermore, there is a record of messages kept in the workflow for the llm to recall previous messages.

I use Ollama llama3.2 for embedding and language models, and the user can use any Ollama model of their choice as long as they download it beforehand.

### Next Steps

I would like to evaluate the RAG performance when adjusting key parameters in the workflow, namely the document chunk size, document overlap, value of k relavent documents, and semantic search algorithm. These parameters can drastically affect the context and quality of the augmented llm prompt. I would also like to test different Ollama models, and am currently extending to HuggingFace open source models for biomedical fine-tuned llms.

## Installation

Prerequisites include:
- [anaconda](https://anaconda.org/)/[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) for environment setup. 
- [Ollama](https://ollama.com/) for local llms

```bash
conda env create -f install/environment.yml
```

## Usage



The main script takes in the following inputs:
- `search_topic`: a phrase for which you would like to find pubmed articles for
- `max_results`: maximum number of articles to fetch relating to your topipc
- `llm_model`: name of ollama llm of your choice
- `embedding_model`: name of ollama embedding model of your choice

Example:
You are a bionformatics researcher focused on developing computational approaches for single cell omics data. You want to test your tool out on Alzheimer's disease case but you are not an expert in this disease. You can use this RAG system to search for this disease to get started on your research.

```python
python pubmed_rag_ollama.py --search_topic "Alzheimer's disease scRNAseq" --max_results 50 --llm_model llama3.2 --embedding_model=llama3.2
```

You are a wet lab researcher seeking to explore gene network analysis on your dataset and would like to know how they generally work. You can use this RAG system to give a general overview of the methdologies and workflows so you can better understand the results from your analysis.

```python
python pubmed_rag_ollama.py --search_topic "scRNAseq gene regulatory networks" --max_results 50 --llm_model llama3.2 --embedding_model=llama3.2
```