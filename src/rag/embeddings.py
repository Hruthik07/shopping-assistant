"""Embedding generation for RAG."""

from typing import List
from src.utils.config import settings
from src.analytics.logger import logger
from src.utils.cache import cache_service


class EmbeddingGenerator:
    """Generate embeddings for text."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.use_openai = settings.use_openai_embeddings

        if self.use_openai:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=settings.openai_api_key
            )
        else:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text (with caching)."""
        try:
            # Check cache first
            cached_embedding = await cache_service.get_embedding(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return cached_embedding

            # Generate embedding
            if self.use_openai:
                embedding = self.embeddings.embed_query(text)
            else:
                embedding = self.model.encode(text).tolist()

            # Cache the embedding
            await cache_service.set_embedding(
                text=text, embedding=embedding, ttl=settings.cache_embedding_ttl
            )

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if self.use_openai:
                return self.embeddings.embed_documents(texts)
            else:
                return self.model.encode(texts).tolist()
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise


# Global embedding generator
embedding_generator = EmbeddingGenerator()
