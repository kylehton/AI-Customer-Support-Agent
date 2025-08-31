
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import logging
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from configs.models import SupportQuery, SupportResponse, HealthResponse
from configs.database import db_manager
from agentWorkflow.agentSystem import agent_system

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Three-Agent Support System",
    description="AI-powered customer support with Triage, Technical Expert, and Communication Specialist agents",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Three-Agent Support System")
    
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        raise RuntimeError("Invalid configuration")
    
    # Initialize database connection
    db_connected = await db_manager.connect()
    if db_connected:
        logger.info("Database initialization successful")
    else:
        logger.warning("Database initialization failed - running with mock data")
    
    logger.info("Application startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Three-Agent Support System")
    await db_manager.disconnect()
    logger.info("Application shutdown completed")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Three-Agent Support System API",
        "description": "AI-powered customer support with three specialized agents",
        "endpoints": {
            "POST /support-query": "Submit a customer support query",
            "GET /health": "Health check",

        },
    }


@app.post("/support-query", response_model=SupportResponse, tags=["Support"])
async def handle_support_query(query: SupportQuery):
    """
    Main endpoint for processing customer support queries
    
    This endpoint processes customer queries through three AI agents:
    1. Triage Specialist - Reformulates vague queries into technical questions
    2. Technical Expert - Searches knowledge base and creates draft solutions
    3. Communication Specialist - Refines technical drafts into friendly responses
    """
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received support query: '{query.query}'")
        
        # Process query through three-agent system
        result = await agent_system.process_support_query(query.query)
        
        response = SupportResponse(
            final_answer=result["final_answer"],
            sources=result["sources"]
        )
        
        logger.info(f"Successfully processed query with {len(response.sources)} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query '{query.query}': {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request. Please try again later."
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():

    try:
        # Check agent status
        agent_status = await agent_system.health_check()
        
        # Check database status
        doc_count = await db_manager.get_document_count()
        database_status = "connected" if db_manager.is_connected else "disconnected"
        
        response = HealthResponse(
            status="healthy",
            timestamp = datetime.now(timezone.utc),
            agents=agent_status
        )
        
        logger.info(f"Health check: {response.status}, DB: {database_status}, Docs: {doc_count}")
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp = datetime.now(timezone.utc),
            agents={
                "triage_specialist": "error",
                "technical_expert": "error",
                "communication_specialist": "error"
            }
        )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "GET /",
                "POST /support-query",
                "GET /health",
                "GET /stats",
                "GET /docs"
            ]
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "The request data is invalid",
            "details": exc.detail if hasattr(exc, 'detail') else str(exc)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL
    )