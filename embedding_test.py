from sentence_transformers import SentenceTransformer

# Test texts (similar to your markets)
texts = [
    "Will Novak Djokovic win the 2026 Australian Open?",
    "Djokovic to win Australian Open 2026",
    "Will Bitcoin reach $100,000 by March 2026?",
    "Will the Lakers win the NBA Championship?",
]

for model_name in ["all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]:
    print(f"\n{model_name}:")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts)
    
    # Normalize and compute similarity
    from numpy import dot
    from numpy.linalg import norm
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = dot(emb[i], emb[j]) / (norm(emb[i]) * norm(emb[j]))
            print(f"  {texts[i][:30]}... vs {texts[j][:30]}... = {sim:.3f}")