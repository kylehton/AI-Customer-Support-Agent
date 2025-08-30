
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class SupportQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Customer support query")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class SupportResponse(BaseModel):
    final_answer: str = Field(..., description="Final customer-friendly answer")
    sources: List[str] = Field(..., description="List of source documents used")
    
    class Config:
        schema_extra = {
            "example": {
                "final_answer": "I'm sorry to hear you're having trouble connecting your drone. Let's try a few simple steps to fix that...",
                "sources": [
                    "To connect your drone to the mobile app: 1. Ensure drone is powered on...",
                    "If experiencing connectivity issues: First, power cycle both devices..."
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
                "draft_solution": "Solution: 1. Power cycle drone. 2. Verify Bluetooth enabled. 3. Open app settings...",
                "sources": [
                    "Troubleshooting guide section 1.4: Power cycle procedures...",
                    "User manual section 3.2: Bluetooth pairing instructions..."
                ]
            }
        }


class KnowledgeBaseDocument(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source of the document")
    category: str = Field(default="general", description="Document category")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "content": "To connect your drone to the mobile app, follow these steps...",
                "source": "DroneX Pro Manual Section 3.2",
                "category": "connectivity",
                "created_at": "2024-01-01T12:00:00Z"
            }
        }


class SearchResult(BaseModel):
    content: str
    similarity: float
    source: str
    category: Optional[str] = None