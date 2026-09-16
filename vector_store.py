import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self):
        """
        Load the embedding model.
        """

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.index = None
        self.documents = []

    def create_index(self, chunks):
        """
        Create FAISS vector index from chunks.
        """

        self.documents = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

    def search(self, query, k=4):
        """
        Search the vector database.
        """

        if self.index is None:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            min(k, len(self.documents))
        )

        results = []

        for index in indices[0]:

            if index != -1:

                results.append(
                    self.documents[index]
                )

        return results