from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import logging
import uvicorn
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from configs.models import SupportQuery, SupportResponse, HealthResponse
from configs.database import db_manager
from agent.agentic_system import agentic_system
from agent.agent_tools import SUPPORT_TOOLS


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Support System",
    description="AI-powered customer support with fully autonomous agents using LangGraph",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Agentic Support System with LangGraph")
    
    # Validate configuration
    if not hasattr(config, 'validate') or not config.validate():
        logger.error("Configuration validation failed")
        raise RuntimeError("Invalid configuration")
    
    # Initialize database connection
    try:
        db_connected = await db_manager.connect()
        if db_connected:
            logger.info("Database initialization successful")
        else:
            logger.warning("Database initialization failed - running with mock data")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
    
    # Test agentic system initialization
    try:
        health = await agentic_system.health_check()
        if health.get("workflow") == "compiled":
            logger.info("Agentic workflow successfully compiled and ready")
        else:
            logger.warning("Agentic workflow initialization issues detected")
    except Exception as e:
        logger.warning(f"Agentic system health check failed: {e}")
    
    logger.info("Application startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Agentic Support System")
    try:
        await db_manager.disconnect()
    except Exception as e:
        logger.error(f"Error during database disconnect: {e}")
    logger.info("Application shutdown completed")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Agentic Support System API",
        "description": "AI-powered customer support with fully autonomous agents using LangGraph",
        "features": [
            "Autonomous agent workflow",
            "Dynamic tool selection",
            "Intelligent query processing",
            "Vector knowledge base search",
            "Customer-friendly response generation"
        ],
        "endpoints": {
            "POST /support-query": "Submit a customer support query to the agentic system",
            "GET /health": "Health check for agents and system components",
        },
    }


@app.post("/support-query", response_model=SupportResponse, tags=["Support"])
async def handle_support_query(query: SupportQuery):
    """
    Main endpoint for processing customer support queries through the agentic system.
    
    The agentic system uses LangGraph to orchestrate autonomous agents that:
    1. Analyze and reformulate customer queries
    2. Search the knowledge base using vector similarity
    3. Generate technical solutions from source materials
    4. Create customer-friendly responses
    
    The agents work autonomously, making decisions about which tools to use and when,
    based on the current state of the conversation and available information.
    """
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received agentic support query: '{query.query}'")
        
        # Process query through agentic system
        result = await agentic_system.process_support_query(query.query)
        
        response = SupportResponse(
            final_answer=result["final_answer"],
            sources=result["sources"]
        )
        
        logger.info(f"Agentic system successfully processed query with {len(response.sources)} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing agentic query '{query.query}': {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request. Please try again later."
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for the agentic support system."""
    try:
        # Check agentic system status
        agent_status = await agentic_system.health_check()
        
        # Check database status
        try:
            doc_count = await db_manager.get_document_count()
            database_status = "connected" if hasattr(db_manager, 'is_connected') and db_manager.is_connected else "disconnected"
        except Exception as e:
            doc_count = 0
            database_status = "error"
            logger.warning(f"Database health check failed: {e}")
        
        # Determine overall status
        overall_status = "healthy"
        if agent_status.get("workflow") != "compiled":
            overall_status = "degraded"
        if database_status != "connected":
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            agents=agent_status
        )
        
        logger.info(f"Agentic health check: {response.status}, DB: {database_status}, Docs: {doc_count}")
        return response
        
    except Exception as e:
        logger.error(f"Agentic health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc),
            agents={
                "coordinator": "error",
                "tool_executor": "error",
                "finalizer": "error",
                "workflow": "error"
            }
        )


@app.get("/workflow-info", tags=["System"])
async def workflow_info():
    """Get information about the agentic workflow and available tools."""
    try:
        
        tools_info = []
        for tool in SUPPORT_TOOLS:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": getattr(tool.args_schema, '__name__', 'BaseModel') if hasattr(tool, 'args_schema') else "N/A"
            })
        
        return {
            "workflow_type": "LangGraph State Machine",
            "total_tools": len(SUPPORT_TOOLS),
            "tools": tools_info,
            "workflow_steps": [
                {
                    "step": "coordinator",
                    "description": "Orchestrates the workflow and decides which tools to use"
                },
                {
                    "step": "tool_executor", 
                    "description": "Executes the selected tools with appropriate parameters"
                },
                {
                    "step": "finalizer",
                    "description": "Prepares the final response for the customer"
                }
            ],
            "process_flow": [
                "1. Triage customer query into technical terms",
                "2. Search knowledge base for relevant sources",
                "3. Generate technical draft solution",
                "4. Convert to customer-friendly response"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving workflow information")


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
                "GET /workflow-info",
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
            "message": "An unexpected error occurred in the agentic system. Please try again later."
        }
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=getattr(config, 'HOST', '0.0.0.0'),
        port=getattr(config, 'PORT', 8000),
        reload=getattr(config, 'RELOAD', True),
        log_level=getattr(config, 'LOG_LEVEL', 'info').lower()
    )