import lancedb
from sentence_transformers import SentenceTransformer
import bm25s
import pickle
import argparse
from rerankers import Reranker

# Load pre-computed resources
def load_resources(db_path="blog_search.db", table_name="blog_chunks", 
                  model_name="sentence-transformers/all-MiniLM-L6-v2",
                  bm25_corpus_path="bm25_corpus.pkl"):
    """Load pre-computed resources"""
    model = SentenceTransformer(model_name)
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    df = table.to_pandas()
    
    # BM25 corpus
    with open(bm25_corpus_path, 'rb') as f:
        bm25_data = pickle.load(f)
        corpus, corpus_tokens= bm25_data['corpus'], bm25_data['corpus_tokens']
    
    # BM25 retriever
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    
    #  eranker
    ranker = Reranker('cross-encoder', model='mixedbread-ai/mxbai-rerank-base-v1', verbose=False)
    
    return model, table, df, retriever, ranker

# Load resources once at module level
model, table, df, retriever, ranker = load_resources()

def vector_search(query):
    """Perform vector search"""
    query_embedding = model.encode(query, normalize_embeddings=True)
    vector_results = table.search(query_embedding).metric('cosine').to_pandas()
    vector_results['vector_score'] = 1 - vector_results['_distance']
    return vector_results

def bm25_search(query, vector_results):
    """Perform keyword search using BM25"""
    # Tokenize the query and retrieve results
    query_tokens = bm25s.tokenize(query)
    docs, scores = retriever.retrieve(query_tokens, k=len(df))
    
    # Map BM25 scores to our dataframe indices
    bm25_scores = {i: scores[0, idx] for idx, i in enumerate(docs[0])}
    vector_results['bm25_score'] = vector_results.index.map(
        lambda x: bm25_scores.get(x, 0) if x in bm25_scores else 0)
    
    # Normalize BM25 scores
    if vector_results['bm25_score'].max() > 0:
        vector_results['bm25_score'] = vector_results['bm25_score'] / vector_results['bm25_score'].max()
    
    return vector_results

def hybrid_search(query, top_k=5, vector_weight=0.7):
    """Perform hybrid search combining vector and keyword search"""
    # Get results
    vector_results = vector_search(query)
    bm25_results = bm25_search(query, vector_results)
    
    # Combine scores with weighting
    bm25_results['combined_score'] = (
        vector_weight * bm25_results['vector_score'] + 
        (1 - vector_weight) * bm25_results['bm25_score'])
    
    return bm25_results.sort_values('combined_score', ascending=False).head(top_k)

def rerank_search(candidates, query):
    """Rerank candidates using cross-encoder"""
    texts = candidates['content'].tolist()
    doc_ids = candidates.index.tolist()
    ranked = ranker.rank(query=query, docs=texts, doc_ids=doc_ids)
    return ranked

def search_blog_posts(query, top_k=3):
    """Search blog posts using hybrid search followed by cross-encoder reranking"""
    # Get candidates with hybrid search
    candidates = hybrid_search(query, top_k=top_k*2)
    
    # Rerank candidates
    ranked = rerank_search(candidates, query)
    
    # Map scores back to candidates and return top results
    candidates['rerank_score'] = candidates.index.map(
        {r.document.doc_id: r.score for r in ranked.results}.get)
    return candidates.sort_values('rerank_score', ascending=False).head(top_k)

def format_results(results, score_col='rerank_score'):
    """Format search results for display"""
    output = []
    
    for _, row in results.iterrows():
        output.append(f"\n{'='*80}\n")
        output.append(f"Post: {row['post_title']}")
        output.append(f"Section: {row['chunk_title']}")
        output.append(f"Score: {row[score_col]:.4f}")
        output.append(f"\n{'-'*40}\n")
        
        # Show a snippet of the content (first 300 chars)
        content_snippet = row['content'][:300] + "..." if len(row['content']) > 300 else row['content']
        output.append(content_snippet)
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Search blog posts")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--method", choices=["hybrid", "vector", "keyword", "rerank"], 
                        default="rerank", help="Search method")
    
    args = parser.parse_args()
    
    # Perform search based on method
    if args.method == "vector":
        results = vector_search(args.query)
        results = results.sort_values('vector_score', ascending=False).head(args.top_k)
        score_col = 'vector_score'
    elif args.method == "keyword":
        results = bm25_search(args.query, vector_search(args.query))
        results = results.sort_values('bm25_score', ascending=False).head(args.top_k)
        score_col = 'bm25_score'
    elif args.method == "hybrid":
        results = hybrid_search(args.query, args.top_k)
        score_col = 'combined_score'
    else:  # rerank
        results = search_blog_posts(args.query, args.top_k)
        score_col = 'rerank_score'
    
    print(f"\nSearch results for: '{args.query}'")
    print(f"Method: {args.method}")
    print(format_results(results, score_col))

if __name__ == "__main__":
    main() 