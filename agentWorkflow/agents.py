from typing import List
from openai import OpenAI
import logging

from configs.config import config
from configs.database import db_manager
from configs.models import TechnicalExpertResponse
import os

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY 

if config.OPENAI_API_KEY:
    logger.info("OpenAI API initialized")
    client = OpenAI()
else:
    logger.warning("OpenAI API key not found. Using mock responses for demonstration.")


class Agent1_TriageSpecialist:
    
    def __init__(self):
        self.name = "Triage Specialist"
        self.system_prompt = """
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
        """
    
    async def process(self, customer_query: str) -> str:
   
        try:
            if config.OPENAI_API_KEY:
                response = await self._call_openai(customer_query)
            else:
                response = self._mock_response(customer_query)
            
            logger.info(f"Triage Specialist processed: '{customer_query}' → '{response}'")
            return response
        except Exception as e:
            logger.error(f"Error in Triage Specialist: {e}")
            return self._fallback_response(customer_query)
    
    async def _call_openai(self, query: str) -> str:

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Reformulate this customer query into a technical search query: '{query}'"}
            ]
            
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.AGENT_TEMPERATURE
            )
            
            logger.info("Agent 1:", response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error in Triage Specialist: {e}")
            return "OpenAI API error in Triage Specialist Agent"
    
    
    def _fallback_response(self, query: str) -> str:
        return f"Technical support query regarding: {query}"

class Agent2_TechnicalExpert:

    def __init__(self):
        self.name = "Technical Expert"
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
        """

    async def process(self, technical_query: str) -> TechnicalExpertResponse:
        sources: List[str] = []
        try:
            # Search knowledge base
            search_results = await db_manager.search_documents(technical_query, config.MAX_SOURCES)

            if not search_results:
                logger.warning(f"No sources found for query: {technical_query}")
                return TechnicalExpertResponse(
                    draft_solution="Unable to find specific information in the knowledge base. Please contact technical support for further assistance.",
                    sources=[]
                )

            # Extract source content
            sources = [result.content for result in search_results]

            # Generate draft solution using OpenAI
            draft_solution = await self._call_openai(technical_query, sources) if config.OPENAI_API_KEY else ""

            logger.info(f"Technical Expert generated draft with {len(sources)} sources")
            return TechnicalExpertResponse(
                draft_solution=draft_solution,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Error in Technical Expert: {e}")
            fallback_text = self._fallback_response(technical_query, sources)
            return TechnicalExpertResponse(
                draft_solution=fallback_text,
                sources=sources
            )

    async def _call_openai(self, query: str, sources: List[str]) -> str:
        try:
            sources_text = "\n\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                Based on the following sources, create a technical draft solution for: '{query}'

                Available Sources:
                {sources_text}

                Provide a clear, step-by-step solution based only on the information in these sources.
                """}
            ]

            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.AGENT_TEMPERATURE
            )

            logger.info("Agent 2:", response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error in Technical Expert: {e}")
            return self._fallback_solution(query, sources)

    def _fallback_response(self, query: str, sources: List[str]) -> str:
        return (
            f"An error occurred while generating a technical draft for your query: '{query}'. "
            f"Please follow general troubleshooting steps or contact technical support. "
            f"Sources found: {len(sources)}"
        )


class Agent3_CommunicationSpecialist:
    
    def __init__(self):
        self.name = "Communication Specialist"
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
        all important technical details.
        """
    
    async def process(self, original_query: str, draft_solution: str, sources: List[str]) -> str:

        try:
            if config.OPENAI_API_KEY:
                response = await self._call_openai(original_query, draft_solution, sources)
            else:
                response = self._mock_response(original_query, draft_solution)

            logger.info("Communication Specialist generated customer-friendly response")
            return response
        except Exception as e:
            logger.error(f"Error in Communication Specialist: {e}")
            return self._fallback_response(original_query, draft_solution)

    async def _call_openai(self, query: str, draft_solution: str, sources: List[str]) -> str:

        try:
            sources_text = "\n\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources)])

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                The customer asked: '{query}'

                Here is the technical draft solution:
                {draft_solution}

                Available sources:
                {sources_text}

                Please rewrite this solution into a warm, empathetic, and easy-to-follow response for the customer. Always respond in plain text only. Do not use markdown, headers, or lists.
                """}
            ]

            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.AGENT_TEMPERATURE
            )

            logger.info("Agent 3:", response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error in Communication Specialist: {e}")
            return "OpenAI API error in Communication Specialist Agent"

    def _mock_response(self, query: str, draft_solution: str) -> str:

        return (
            f"I understand you're having trouble with: {query}. "
            f"Here's a simplified explanation of the steps: {draft_solution}. "
            "If you run into any issues, please let us know — we're happy to help further!"
        )

    def _fallback_response(self, query: str, draft_solution: str) -> str:
        return f"Customer support response regarding: {query}. Suggested steps: {draft_solution}"
