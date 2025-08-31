from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone


class SupportQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Customer support query")



class SupportResponse(BaseModel):
    final_answer: str = Field(..., description="Final customer-friendly answer")
    sources: List[str] = Field(..., description="List of source documents used")
    
    class Config:
        schema_extra = {
            "example": {
                "final_answer": "I'm sorry to hear you're having trouble with cloning a repository. Let's try a few simple steps to fix that...",
                "sources": [
                    "To clone a repository: 1. Ensure you have the repository open...",
                    "If experiencing cloning issues: First, check your device SSH key has been entered into GitHub..."
                ]
            }
        }


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    agents: Dict[str, str]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "agents": {
                    "triage_specialist": "active",
                    "technical_expert": "active",
                    "communication_specialist": "active"
                }
            }
        }


class AgentResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None


class TechnicalExpertResponse(BaseModel):
    draft_solution: str
    sources: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "draft_solution": "1. Install Git locally . . . 2. Configure SSH key . . . 3. Clone repository to local machine . . ",
                "sources": [
                    "Troubleshooting guide section 1.4: Shell Connections",
                    "User manual section 3.2: Cloning repositories"
                ]
            }
        }


class KnowledgeBaseDocument(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source of the document")
    category: str = Field(default="general", description="Document category")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    created_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    
    class Config:
        schema_extra = {
            "example": {
                "content": "To connect your repository, follow these steps...",
                "source": "GitHub Repository FAQ",
                "category": "repositories",
                "created_at": "2025-08-30T12:00:00Z"
            }
        }


class SearchResult(BaseModel):
    content: str
    similarity: float
    source: str
    category: Optional[str] = None