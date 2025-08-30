"""
Database connection and operations for AI Customer Support
"""

import sys
import os
from typing import List, Optional
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import config
from configs.models import SearchResult

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.is_connected: bool = False

    async def connect(self) -> bool:
        """Connect to MongoDB and initialize embedding model."""
        try:
            self.client = AsyncIOMotorClient(config.MONGODB_URL)
            self.db = self.client[config.DATABASE_NAME]
            self.collection = self.db[config.COLLECTION_NAME]

            # Test connection
            await self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info(f"Successfully loaded embedding model: {config.EMBEDDING_MODEL}")

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.is_connected = False

            # Still try to load embedding model for mock responses
            try:
                self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
                logger.info("Loaded embedding model for mock responses")
            except Exception as embed_error:
                logger.error(f"Failed to load embedding model: {embed_error}")

            return False

    async def disconnect(self):
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Database connection closed")

    async def add_document(self, content: str, source: str, category: str = "general") -> bool:
        """Add a document with embedding to MongoDB."""
        if not self.is_connected or self.collection is None:
            logger.warning("Database not connected, cannot add document")
            return False

        try:
            embedding = self.embedding_model.encode([content])[0].tolist()
            document = {
                "content": content,
                "embedding": embedding,
                "source": source,
                "category": category,
                "created_at": datetime.now(timezone.utc)
            }
            result = await self.collection.insert_one(document)
            logger.info(f"Added document with ID: {result.inserted_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False

    async def search_documents(self, query: str, max_results: int = None) -> List[SearchResult]:
        """Search MongoDB for documents similar to the query."""
        if max_results is None:
            max_results = config.MAX_SOURCES

        if not self.is_connected or self.collection is None:
            logger.warning("Database not connected, returning mock results")
            return []

        try:
            query_embedding = self.embedding_model.encode([query])[0]
            cursor = self.collection.find({}, {"content": 1, "embedding": 1, "source": 1, "category": 1})
            results = []

            async for doc in cursor:
                if "embedding" in doc and doc["embedding"]:
                    doc_embedding = np.array(doc["embedding"])
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    if similarity >= config.SIMILARITY_THRESHOLD:
                        results.append(SearchResult(
                            content=doc["content"],
                            similarity=float(similarity),
                            source=doc.get("source", "Unknown"),
                            category=doc.get("category", "general")
                        ))

            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:max_results]

        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []

    async def get_document_count(self) -> int:
        """Return total number of documents in the collection."""
        if not self.is_connected or self.collection is None:
            return 0
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


# Global database manager instance
db_manager = DatabaseManager()
