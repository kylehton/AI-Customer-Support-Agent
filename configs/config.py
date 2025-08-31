import os
from dotenv import load_dotenv

load_dotenv()

class Config:    
    # Database settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "support_kb")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RAG settings
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    #CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    #CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Agent settings
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    
    @classmethod
    def validate(cls) -> bool:
        try:
            # Check required settings
            if not cls.MONGODB_URL:
                raise ValueError("MONGODB_URL is required")
            
            if cls.MAX_SOURCES <= 0:
                raise ValueError("MAX_SOURCES must be positive")
            
            if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
                raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Create global config instance
config = Config()