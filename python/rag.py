import faiss
import numpy as np

class RAGIndexer:
    def __init__(self, emb_dim=768):
        self.index = faiss.IndexFlatL2(emb_dim)
        self.chunks = []

    def add_chunks(self, chunks, embeddings):
        # embeddings: np.array of shape (N, 768)
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)

    def retrieve(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]
