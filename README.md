# RAG Evaluation Demo App

A toolkit for building, searching, and evaluating retrieval-augmented generation (RAG) systems using blog posts from isaacflath.com.  This is intended as a demo app for educational purposes.

This is a companion repo for the following blog posts:
- [Retrieval 101](https://isaacflath.com/blog/blog_post?fpath=posts%2F2025-03-17-Retrieval101.ipynb)
- [Building a Search Evaluation App with FastHTML and MonsterUI](https://isaacflath.com/blog/blog_post?fpath=posts%2F2025-03-18-Builing-a-search-eval-app-fasthtml-monsterui.ipynb)

## Overview

This project provides tools to:

1. Create search indices from blog post content (lancedb vectors and bm25 corpus)
2. Search blog posts using various retrieval methods
3. Evaluate and compare search results through an interactive web application

> Note:  This is deployed to railway [here](https://search-eval-demo-production.up.railway.app/).  But it's not designed for multiple users as it's a demo so if too many people are there it'll probably be super slow!
> 
## Components

### 1. Evaluation Application

The `main.py` script provides a web interface for testing and evaluating search results:

```
python main.py
```

Features:
- Interactive search interface with multiple retrieval methods
- Relevance rating system (1-5 scale)
- Notes and annotations for search results
- Historical evaluation tracking and comparison

### 2. Content Storage

- `rendered_posts/`: Contains blog posts from isaacflath.com in both HTML and Markdown formats.  `create_search_index.py` uses these, and they are provided for your own experimentation.

### 3. Search Index Creation

The `create_search_index.py` script processes blog posts and creates search indices:

```
python create_search_index.py [--rerun]
```

Options:
- `--rerun`: Force regeneration of embeddings and indices (otherwise skips if they already exist)

Outputs:
- `blog_search.db`: LanceDB database containing vector embeddings
- `bm25_corpus.pkl`: Pickled corpus for BM25 keyword search

### 4. Search Functionality

The `search_blog.py` script provides a command-line interface for searching blog posts:

```
python search_blog.py "your search query" [--top-k N] [--method METHOD]
```

Options:
- `--top-k N`: Number of results to return (default: 3)
- `--method METHOD`: Search method to use (choices: vector, keyword, hybrid, rerank; default: rerank)

Search methods:
- `vector`: Dense retrieval using sentence embeddings
- `keyword`: Sparse retrieval using BM25 algorithm
- `hybrid`: Combined vector and keyword search
- `rerank`: Two-stage retrieval with cross-encoder reranking



## Getting Started

1. Clone the repository
2. Install dependencies `pip install -r requiements.txt`
3. Use python `main.py` and navigate to `localhost:5001` to run the app.
4. Run `create_search_index.py` to build search indices
5. Use `search_blog.py` for command-line searching
