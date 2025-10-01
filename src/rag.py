#!/usr/bin/env python3
"""
RAG demo using semantic embeddings + FAISS (if available), with a numpy fallback.
- Builds sentence-level embeddings for docs in ../docs
- Indexes them in FAISS (or in-memory numpy)
- Retrieves top-k and assembles a short extractive answer

Requirements (for full experience):
  pip install sentence-transformers faiss-cpu
If faiss is not installed, this script falls back to a cosine-similarity search with numpy.
"""
import os, glob, numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

def load_docs(docs_dir):
    texts = []
    for p in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def split_sentences(text):
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if s.strip()]
    return sents

def build_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    # If sentence-transformers is not available, fall back to a simple TF-IDF vectorization (not semantic)
    if SentenceTransformer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english").fit(sentences)
        mat = vec.transform(sentences).toarray()
        return mat, None  # second return None indicates no semantic model
    model = SentenceTransformer(model_name)
    emb = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    return emb, model

def build_index(embeddings):
    # Try to use FAISS if available
    try:
        import faiss
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        # normalize for cosine similarity when using inner product
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return ("faiss", index)
    except Exception as e:
        # fallback to numpy-based storage (will compute cosine sims at query time)
        return ("numpy", embeddings)

def retrieve(query, sentences, embeddings, model=None, index=None, topk=3):
    # create query embedding
    if model is None and SentenceTransformer is not None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if model is not None:
        q_emb = model.encode([query], convert_to_numpy=True)
        q_emb = q_emb.astype('float32')
    else:
        # fallback: use simple tf-idf vectorizer for query; embeddings is actually TF-IDF matrix then
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english").fit(sentences)
        q_emb = vec.transform([query]).toarray()

    if isinstance(index, tuple) and index[0] == "numpy":
        # numpy fallback: embeddings is matrix, compute cosine sim
        mat = index[1] if len(index)>1 else embeddings
        # normalize
        def _norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n[n==0] = 1.0
            return a / n
        M = _norm(mat)
        qn = q_emb / (np.linalg.norm(q_emb)+1e-12)
        sims = (M @ qn.T).ravel()
        idx = np.argsort(-sims)[:topk]
        return [(i, sims[i]) for i in idx if sims[i] > 0]
    else:
        # FAISS index (or index is faiss index)
        try:
            import faiss
            qn = q_emb.astype('float32')
            faiss.normalize_L2(qn)
            D, I = index.search(qn, topk)
            # D contains inner product scores; I indices
            results = [(int(I[0,i]), float(D[0,i])) for i in range(I.shape[1]) if I[0,i] != -1]
            return results
        except Exception as e:
            # fallback to simple numpy cosine if anything fails
            M = embeddings
            def _norm(a):
                n = np.linalg.norm(a, axis=1, keepdims=True)
                n[n==0] = 1.0
                return a / n
            M = _norm(M)
            qn = q_emb / (np.linalg.norm(q_emb)+1e-12)
            sims = (M @ qn.T).ravel()
            idx = np.argsort(-sims)[:topk]
            return [(i, sims[i]) for i in idx if sims[i] > 0]

def answer(query, sentences, embeddings, model, index, topk=5):
    res = retrieve(query, sentences, embeddings, model=model, index=index, topk=topk)
    if not res:
        return "No relevant information found in the local knowledge base."
    out = "Context (top retrieved passages):\\n"
    for rank, (i, score) in enumerate(res, start=1):
        out += f"{rank}. (score={score:.3f}) {sentences[i]}\\n"
    out += "\\nGenerated Answer (extractive + template):\\n"
    out += "Based on the available documentation, here's a concise response:\\n"
    out += " ".join([sentences[i] for (i,_) in res[:3]])
    return out

def build_from_docs(docs_dir):
    texts = load_docs(docs_dir)
    sents = []
    for t in texts:
        sents.extend(split_sentences(t))
    embeddings, model = build_embeddings(sents)
    # If TF-IDF fallback, we return embeddings matrix and model=None
    index = build_index(embeddings)
    return sents, embeddings, model, index

def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    sents, embeddings, model, index = build_from_docs(docs_dir)
    print("RAG (embeddings + FAISS) demo — type questions (Ctrl+C to quit)")
    while True:
        q = input("\\nYour question: ").strip()
        if not q:
            continue
        print("\\n" + answer(q, sents, embeddings, model, index, topk=5) + "\\n")

if __name__ == "__main__":
    main()