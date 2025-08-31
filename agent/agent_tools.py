from typing import List, Dict, Any, Optional, Type
import logging
from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from configs.config import config
from configs.database import db_manager
from configs.models import TechnicalExpertResponse, SearchResult

logger = logging.getLogger(__name__)


class TriageQueryInput(BaseModel):
    """Input for triage query tool."""
    customer_query: str = Field(description="The customer's original query that needs to be triaged")


class KnowledgeSearchInput(BaseModel):
    """Input for knowledge base search tool."""
    technical_query: str = Field(description="Technical query to search in the knowledge base")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return")


class DraftSolutionInput(BaseModel):
    """Input for draft solution generation tool."""
    technical_query: str = Field(description="The technical query")
    sources: List[str] = Field(description="List of source materials")


class CustomerResponseInput(BaseModel):
    """Input for customer response generation tool."""
    original_query: str = Field(description="Original customer query")
    draft_solution: str = Field(description="Technical draft solution")
    sources: List[str] = Field(description="Source materials used")


class TriageQueryTool(BaseTool):
    """Tool for triaging and reformulating customer queries into technical queries."""
    
    name: str = "triage_query"
    description: str = """
    Analyzes vague customer queries and reformulates them into precise, technical questions 
    suitable for searching a knowledge base. Use this when you have a raw customer query 
    that needs to be converted into technical terminology.
    """
    args_schema: Type[BaseModel] = TriageQueryInput

    llm: ChatOpenAI = Field(default=None)
    system_prompt: str = ""
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        ) if config.OPENAI_API_KEY else None
        
        self.system_prompt: str = """
        You are a tier-1 technical support specialist with expertise in consumer electronics, 
        particularly drones, mobile apps, and IoT devices. Your job is to analyze vague customer 
        queries and reformulate them into precise, technical questions suitable for searching 
        a knowledge base.
        
        Key responsibilities:
        1. Identify the core technical issue from customer descriptions
        2. Add relevant technical keywords and terminology
        3. Expand abbreviations and clarify ambiguous terms
        4. Structure the query for effective knowledge base searching
        
        Examples:
        - "My drone isn't working" → "Troubleshooting steps for drone power issues, connectivity problems, and basic functionality failures"
        - "App won't connect" → "Mobile application connectivity issues, pairing problems, Bluetooth and Wi-Fi troubleshooting"
        - "Battery problems" → "Battery charging issues, power management, battery life optimization, and replacement procedures"
        
        Always focus on actionable technical aspects that can be found in product manuals.
        Return only the reformulated technical query, nothing else.
        """

    def _run(
        self, 
        customer_query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the triage query tool."""
        try:
            if self.llm:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"Reformulate this customer query into a technical search query: '{customer_query}'")
                ]
                
                response = self.llm.invoke(messages)
                result = response.content.strip()
                
                logger.info(f"Triage tool processed: '{customer_query}' → '{result}'")
                return result
            else:
                # Fallback without LLM
                result = f"Technical support query regarding: {customer_query}"
                logger.warning(f"Triage tool using fallback for: '{customer_query}'")
                return result
                
        except Exception as e:
            logger.error(f"Error in triage tool: {e}")
            return f"Technical support query regarding: {customer_query}"


class KnowledgeSearchTool(BaseTool):
    """Tool for searching the knowledge base using vector similarity."""
    
    name: str = "search_knowledge_base"
    description: str = """
    Searches the knowledge base using vector similarity to find relevant documentation 
    and troubleshooting information. Use this when you have a technical query and need 
    to find relevant source materials.
    """
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    def _run(
        self, 
        technical_query: str, 
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Execute the knowledge base search."""
        try:
            # Use asyncio to run the async database search
            import asyncio
            loop = asyncio.get_event_loop()
            
            search_results: List[SearchResult] = loop.run_until_complete(
                db_manager.search_documents(technical_query, max_results=max_results)
            )
            
            if not search_results:
                logger.warning(f"No sources found for query: {technical_query}")
                return {
                    "sources": [],
                    "source_count": 0,
                    "query": technical_query
                }
            
            sources = [result.content for result in search_results]
            
            logger.info(f"Knowledge search found {len(sources)} sources for: '{technical_query}'")
            
            return {
                "sources": sources,
                "source_count": len(sources),
                "query": technical_query
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge search tool: {e}")
            return {
                "sources": [],
                "source_count": 0,
                "query": technical_query,
                "error": str(e)
            }


class DraftSolutionTool(BaseTool):
    """Tool for generating technical draft solutions from source materials."""
    
    name: str = "generate_draft_solution"
    description: str = """
    Synthesizes information from knowledge base sources into a technical draft solution.
    Use this when you have relevant source materials and need to create a structured
    technical response.
    """
    args_schema: Type[BaseModel] = DraftSolutionInput

    llm: ChatOpenAI = Field(default=None)
    system_prompt: str = ""
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        ) if config.OPENAI_API_KEY else None
        
        self.system_prompt = """
        You are a technical expert specializing in consumer electronics support. Your job is to 
        synthesize information from product manuals and knowledge base articles into direct, 
        factual draft solutions.

        Guidelines:
        1. Use only the provided source material
        2. Create step-by-step solutions when appropriate
        3. Include specific section references when available
        4. Be precise and technical but clear
        5. Focus on actionable troubleshooting steps

        Format your response as a structured solution with numbered steps when possible.
        Return only the technical solution, nothing else.
        """

    def _run(
        self, 
        technical_query: str, 
        sources: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate a technical draft solution."""
        try:
            if not sources:
                return "Unable to find specific information in the knowledge base. Please contact technical support for further assistance."
            
            if self.llm:
                sources_text = "\n\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])
                
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"""
                    Based on the following sources, create a technical draft solution for: '{technical_query}'

                    Available Sources:
                    {sources_text}

                    Provide a clear, step-by-step solution based only on the information in these sources.
                    """)
                ]
                
                response = self.llm.invoke(messages)
                result = response.content.strip()
                
                logger.info(f"Draft solution generated for: '{technical_query}'")
                return result
            else:
                # Fallback without LLM
                result = f"Technical draft for: {technical_query}. Based on {len(sources)} sources."
                logger.warning(f"Draft solution using fallback for: '{technical_query}'")
                return result
                
        except Exception as e:
            logger.error(f"Error in draft solution tool: {e}")
            return f"Error generating draft solution for: {technical_query}. Please contact support."


class CustomerResponseTool(BaseTool):
    """Tool for converting technical drafts into customer-friendly responses."""
    
    name: str = "generate_customer_response"
    description: str = """
    Transforms technical solutions into friendly, empathetic, and easy-to-follow customer responses.
    Use this when you have a technical draft solution and need to make it customer-friendly.
    """
    args_schema: Type[BaseModel] = CustomerResponseInput

    llm: ChatOpenAI = Field(default=None)

    system_prompt: str = ""
    
    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        ) if config.OPENAI_API_KEY else None
        
        self.system_prompt = """
        You are a customer service communication specialist. Your job is to transform technical 
        solutions into friendly, empathetic, and easy-to-follow customer responses.
        
        Guidelines:
        1. Start with empathy and acknowledgment of the customer's problem
        2. Use friendly, conversational language
        3. Break down technical steps into simple instructions
        4. Add helpful context and encouragement
        5. Maintain accuracy to the source material
        6. End with an offer for additional help
        
        Transform dry technical language into warm, human communication while preserving 
        all important technical details. Always respond in plain text only. Do not use markdown, headers, or lists.
        """

    def _run(
        self, 
        original_query: str, 
        draft_solution: str, 
        sources: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate a customer-friendly response."""
        try:
            if self.llm:
                sources_text = "\n\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])
                
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"""
                    The customer asked: '{original_query}'

                    Here is the technical draft solution:
                    {draft_solution}

                    Available sources:
                    {sources_text}

                    Please rewrite this solution into a warm, empathetic, and easy-to-follow response for the customer.
                    """)
                ]
                
                response = self.llm.invoke(messages)
                result = response.content.strip()
                
                logger.info(f"Customer response generated for: '{original_query}'")
                return result
            else:
                # Fallback without LLM
                result = (
                    f"I understand you're having trouble with: {original_query}. "
                    f"Here's a simplified explanation of the steps: {draft_solution}. "
                    "If you run into any issues, please let us know — we're happy to help further!"
                )
                logger.warning(f"Customer response using fallback for: '{original_query}'")
                return result
                
        except Exception as e:
            logger.error(f"Error in customer response tool: {e}")
            return f"I apologize for the inconvenience with: {original_query}. Please contact our support team for assistance."


# Create tool instances
triage_tool = TriageQueryTool()
search_tool = KnowledgeSearchTool()
draft_tool = DraftSolutionTool()
response_tool = CustomerResponseTool()

# Export tools list
SUPPORT_TOOLS = [
    triage_tool,
    search_tool, 
    draft_tool,
    response_tool
]