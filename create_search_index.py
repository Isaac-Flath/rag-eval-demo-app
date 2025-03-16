import os
import pandas as pd
import lancedb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import bm25s
import argparse
import pickle
from transformers import AutoTokenizer

def read_markdown_files(directory):
    """Read all markdown files from the directory"""
    md_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            
            # Extract title from the first line (markdown header)
            title = content.split('\n')[0].replace('# ', '')
            md_files.append({'title': title, 'content': content, 'path': file_path})
    return md_files

def chunk_by_markdown_sections(markdown_text, min_length=250):
    """Split markdown text into chunks based on header sections."""
    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = []
    current_title = "Introduction"
    
    for line in lines:
        if line.startswith('#'):  # New header found
            # Save previous chunk if it's substantial
            if current_chunk and len('\n'.join(current_chunk)) >= min_length:
                chunks.append({'title': current_title, 'content': '\n'.join(current_chunk)})
            
            # Start new chunk
            current_title = line.lstrip('# ')
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    # Add the final chunk if it exists and meets length requirement
    if current_chunk and len('\n'.join(current_chunk)) >= min_length:
        chunks.append({'title': current_title, 'content': '\n'.join(current_chunk)})
    return chunks

def create_chunks(md_files):
    """Create chunks from markdown files"""
    all_chunks = []
    for md_file in md_files:
        for chunk in chunk_by_markdown_sections(md_file['content']):
            all_chunks.append({'post_title': md_file['title'], 'path': md_file['path'],
                'chunk_title': chunk['title'],'content': chunk['content']})
    return all_chunks

def create_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings for each chunk using SentenceTransformer"""
    model = SentenceTransformer(model_name)
    df = pd.DataFrame(chunks)    
    print("Generating embeddings...")
    
    # Add token count for each chunk
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df['token_count'] = df['content'].apply(lambda x: len(tokenizer.encode(x)))
    
    df['vector'] = df['content'].apply(lambda x: model.encode(x, normalize_embeddings=True))
    
    return df

def store_in_lancedb(df, db_path="blog_search.db", table_name="blog_chunks"):
    """Store the embeddings in LanceDB"""
    db = lancedb.connect(db_path)
    table = db.create_table(table_name, data=df, mode="overwrite")
    table.create_index(vector_column_name="vector")
    return db, table

def save_bm25_corpus(df, output_path="bm25_corpus.pkl"):
    """Save BM25 corpus for later use"""
    corpus = df['content'].tolist()
    corpus_tokens = bm25s.tokenize(corpus)
    with open(output_path, 'wb') as f: pickle.dump({'corpus': corpus, 'corpus_tokens': corpus_tokens}, f)
    print(f"BM25 corpus saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create search index from markdown files")
    parser.add_argument("--rerun", action="store_true", help="Force rerun of embedding generation")
    args = parser.parse_args()
    
    db_path, table_name = "blog_search.db", "blog_chunks"
    # Check if we need to regenerate the index
    db_exists = os.path.exists(db_path)
    bm25_exists = os.path.exists("bm25_corpus.pkl")
    
    if db_exists and bm25_exists and not args.rerun:
        print("Search index already exists. Use --rerun to regenerate.")
        return
    
    print("Reading markdown files...")
    md_files = read_markdown_files(Path('rendered_posts/markdown'))
    print(f"Found {len(md_files)} markdown files")
    
    print("Chunking markdown files...")
    chunks = create_chunks(md_files)
    print(f"Created {len(chunks)} chunks")
    
    df = create_embeddings(chunks)
    
    print("Storing in LanceDB...")
    store_in_lancedb(df, db_path=db_path, table_name=table_name)
    
    print("Saving BM25 corpus...")
    save_bm25_corpus(df)
    
    print(f"Successfully created search index with {len(df)} chunks")
    print(f"Database stored at: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    main()
