## PDF Question Answering System

# Overview

This project demonstrates a PDF Question Answering System using machine learning and natural language processing techniques. The system processes PDF documents into searchable chunks and answers user queries based on the content of the PDFs. Key technologies used include SentenceTransformer, Faiss, Roberta, and TextBlob.

# Features

PDF Loading and Splitting: Load and split PDFs into manageable text chunks.
Embedding Generation: Generate embeddings for text chunks using SentenceTransformer.
Faiss Indexing: Index embeddings for efficient similarity search.
Query Processing: Process user queries to find relevant text chunks.
Answer Extraction: Extract and refine answers using Roberta and TextBlob.

# Technologies Used
SentenceTransformer: For embedding generation
Faiss: For similarity search
Roberta: For question answering
TextBlob: For answer correction
Streamlit: For the user interface

# Installation
Install the required libraries using pip:

pip install langchain sentence-transformers faiss transformers torch textblob streamlit

