# Install required packages
# pip install langchain langchain-community langchain-text-splitters xmltodict faiss-cpu transformers sentence-transformers torch

import os
import xmltodict
import requests
from typing import List, Dict, Any
import time
from bs4 import BeautifulSoup
import argparse


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document
# from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

# initialize global variables
llm=None
vector_db=None
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 1. PubMed and PMC API Utilities

def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "esearchresult" in data and "idlist" in data["esearchresult"]:
        return data["esearchresult"]["idlist"]
    else:
        return []

def fetch_pmids_with_pmcids(pmids: List[str]) -> Dict[str, str]:
    """Fetch PMCIDs corresponding to PMIDs when available.
    Returns a dictionary mapping PMIDs to PMCIDs."""
    if not pmids:
        return {}
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching articles: {response.status_code}")
        return {}
    
    # Parse XML response
    data = xmltodict.parse(response.text)
    
    pmid_to_pmcid = {}
    pubmed_articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
    
    # Handle single article case
    if not isinstance(pubmed_articles, list):
        pubmed_articles = [pubmed_articles]
    
    for article in pubmed_articles:
        pmid = article.get("MedlineCitation", {}).get("PMID", {}).get("#text", "")
        
        # Look for PMCID in PubMed Central IDs
        article_ids = article.get("PubmedData", {}).get("ArticleIdList", {}).get("ArticleId", [])
        if not isinstance(article_ids, list):
            article_ids = [article_ids]
        
        pmcid = ""
        for id_obj in article_ids:
            if id_obj.get("@IdType", "") == "pmc":
                pmcid = id_obj.get("#text", "")
                break
        
        if pmid and pmcid:
            pmid_to_pmcid[pmid] = pmcid
    
    return pmid_to_pmcid

def fetch_pmc_fulltext(pmcid: str) -> str:
    """Fetch the full text content from PubMed Central using the PMCID."""
    if not pmcid:
        return ""
    
    # Strip "PMC" prefix if present
    pmcid = pmcid.replace("PMC", "")
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching PMC full text: {response.status_code}")
        return ""
    
    # Parse the XML to extract the full text
    soup = BeautifulSoup(response.content, 'xml')
    
    # Extract article title
    title = soup.find('article-title')
    title_text = title.text if title else ""
    
    # Extract abstract
    abstract_parts = soup.find_all('abstract')
    abstract_text = ""
    for part in abstract_parts:
        abstract_text += part.text + " "
    
    # Extract full text body
    body = soup.find('body')
    full_text = ""
    
    if body:
        # Extract all paragraphs
        paragraphs = body.find_all('p')
        for p in paragraphs:
            full_text += p.text.strip() + "\n\n"
    
    # Combine all parts
    complete_text = f"Title: {title_text}\n\nAbstract: {abstract_text}\n\nBody: {full_text}"
    return complete_text

def fetch_pubmed_articles_with_fulltext(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch article data including full text when available from PMC."""
    if not pmids:
        return []
    
    # First get the regular PubMed data
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching articles: {response.status_code}")
        return []
    
    # Parse XML response
    data = xmltodict.parse(response.text)
    
    articles = []
    pubmed_articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
    
    # Handle single article case
    if not isinstance(pubmed_articles, list):
        pubmed_articles = [pubmed_articles]
    
    # Get PMCID mappings
    pmid_to_pmcid = fetch_pmids_with_pmcids(pmids)
    
    for article in pubmed_articles:
        article_data = {}
        
        # Extract article metadata
        medline = article.get("MedlineCitation", {})
        article_info = medline.get("Article", {})
        
        # Get PMID
        pmid = medline.get("PMID", {}).get("#text", "")
        article_data["pmid"] = pmid
        
        # Title
        article_data["title"] = article_info.get("ArticleTitle", "")
        
        # Abstract
        abstract_text = article_info.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_text, list):
            abstract_parts = []
            for part in abstract_text:
                if isinstance(part, dict):
                    label = part.get("@Label", "")
                    text = part.get("#text", "")
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                else:
                    abstract_parts.append(part)
            article_data["abstract"] = " ".join(abstract_parts)
        elif isinstance(abstract_text, dict):
            article_data["abstract"] = abstract_text.get("#text", "")
        else:
            article_data["abstract"] = str(abstract_text)
        
        # Journal info
        journal = article_info.get("Journal", {})
        article_data["journal"] = journal.get("Title", "")
        
        # Publication date
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        month = pub_date.get("Month", "")
        day = pub_date.get("Day", "")
        article_data["pub_date"] = f"{year} {month} {day}".strip()
        
        # Authors
        authors = article_info.get("AuthorList", {}).get("Author", [])
        if authors:
            if not isinstance(authors, list):
                authors = [authors]
            
            author_list = []
            for author in authors:
                last_name = author.get("LastName", "")
                fore_name = author.get("ForeName", "")
                if last_name or fore_name:
                    author_list.append(f"{last_name} {fore_name}".strip())
            
            article_data["authors"] = ", ".join(author_list)
        else:
            article_data["authors"] = ""
        
        # Check if full text is available via PMC
        pmcid = pmid_to_pmcid.get(pmid, "")
        article_data["pmcid"] = pmcid
        
        # Try to fetch full text if PMCID is available
        article_data["full_text"] = ""
        if pmcid:
            try:
                print(f"Fetching full text for PMCID: {pmcid}")
                article_data["full_text"] = fetch_pmc_fulltext(pmcid)
                # Add a small delay to be respectful of the NCBI API
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching full text for {pmcid}: {e}")
        
        articles.append(article_data)
    
    return articles

# 2. Document Processing Functions
def articles_to_documents(articles: List[Dict[str, Any]]) -> List[Document]:
    """Convert article dictionaries to LangChain Document objects."""
    documents = []
    
    for article in articles:
        # Determine if we have full text
        has_fulltext = bool(article.get('full_text', ''))
        
        # Create content string
        content = f"Title: {article['title']}\n\n"
        content += f"Authors: {article['authors']}\n\n"
        content += f"Journal: {article['journal']}\n"
        content += f"Publication Date: {article['pub_date']}\n"
        content += f"PMID: {article['pmid']}\n"
        
        if article.get('pmcid'):
            content += f"PMCID: {article['pmcid']}\n\n"
        else:
            content += "\n"
        
        # Add either full text or abstract
        if has_fulltext:
            content += article['full_text']
        else:
            content += f"Abstract: {article['abstract']}"
        
        # Create metadata
        metadata = {
            "title": article['title'],
            "authors": article['authors'],
            "journal": article['journal'],
            "pub_date": article['pub_date'],
            "pmid": article['pmid'],
            "pmcid": article.get('pmcid', ''),
            "has_fulltext": has_fulltext,
            "source": "PubMed/PMC"
        }
        
        # Create Document object
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    return text_splitter.split_documents(documents)

# 3. retrieval tools 
# tool decorator function to retrieve documents.
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve documents related to a query.
    
    Args:
        query: The query to search for.
    
    Returns:
        tuple: string of documents, document objects
    """
    retrieved_docs = vector_db.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
# 3a: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# 3b: Execute the retrieval.
tools = ToolNode([retrieve])

# 3c: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

def create_pubmed_ollama_rag_system(
    search_query: str, 
    max_results: int = 50, 
    llm_model: str = "llama3.2",
    embedding_model: str = "llama3.2"
):
    """
    Create a complete RAG system using Ollama models.
    
    Args:
        search_query (str): PubMed search query
        max_results (int): Maximum number of articles to fetch
        llm_model (str): Ollama LLM model identifier
        embedding_model (str): Ollama embedding model identifier
    
    Returns:
        Dict: RAG system components
    """
    print(f"Searching PubMed for: {search_query}")
    
    # Search and fetch articles (using previous implementation)
    pmids = search_pubmed(search_query, max_results=max_results)
    print(f"Found {len(pmids)} articles")
    
    # Fetch articles with full text when possible
    articles = fetch_pubmed_articles_with_fulltext(pmids)
    
    # Count articles with full text
    fulltext_count = sum(1 for article in articles if article.get('full_text'))
    print(f"Fetched {len(articles)} article details ({fulltext_count} with full text)")
    
    # Process documents
    documents = articles_to_documents(articles)
    split_docs = split_documents(documents)
    print(f"Created {len(split_docs)} document chunks")
    
    # Setup embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Build vector store
    global vector_db
    vector_db = FAISS.from_documents(split_docs, embeddings)
    print("Vector database built successfully")
    
    # Setup LLM
    global llm
    llm = ChatOllama(model=llm_model)
    
    # initialize a graph, to construct the message flow. Each message is a state in a sequence of messages.
    graph_builder = StateGraph(MessagesState)
    # compile all steps into a graph object
    from langgraph.graph import END
    from langgraph.prebuilt import ToolNode, tools_condition

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    print("RAG system ready for queries")
    
    return graph


def answer_query(rag_system, query: str):
    """Answer a query using the RAG system."""
    for step in rag_system.stream(
        {"messages": [{"role": "user", "content": query}]}, 
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    return
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubMed RAG with Ollama models")
    parser.add_argument("--search_topic", type=str, default="scRNAseq methods for gene regulatory network analysis", help="PubMed search topic")
    parser.add_argument("--max_results", type=int, default=50, help="Maximum number of articles to fetch")
    parser.add_argument("--llm_model", type=str, default="llama3.2", help="Ollama LLM model identifier")
    parser.add_argument("--embedding_model", type=str, default="llama3.2", help="Ollama embedding model identifier")
    args = parser.parse_args()

    # Ensure you have enough GPU memory or use a smaller model
    search_topic = args.search_topic
    
    # Create RAG system with Ollama models
    rag_system = create_pubmed_ollama_rag_system(
        search_topic, 
        max_results=args.max_results,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    )
    # input query
    # query = "What are the best methods for analyzing gene regulatory networks using single-cell RNA sequencing data?")
    # answer_query(rag_system, query)
    answer_query(rag_system, "Hello, tell me what reserach articles are you knowledgeable in?")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        answer_query(rag_system, query)
