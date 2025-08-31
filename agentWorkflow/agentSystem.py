from typing import Dict, Any
import logging

from agentWorkflow.agents import Agent1_TriageSpecialist, Agent2_TechnicalExpert, Agent3_CommunicationSpecialist

logger = logging.getLogger(__name__)


class ThreeAgentSupportSystem:
    
    def __init__(self):
        self.agent1 = Agent1_TriageSpecialist()
        self.agent2 = Agent2_TechnicalExpert()
        self.agent3 = Agent3_CommunicationSpecialist()
        
        logger.info("Three-Agent Support System initialized")
    
    async def process_support_query(self, customer_query: str) -> Dict[str, Any]:
    
        logger.info(f"Processing support query: '{customer_query}'")
        
        try:
            # Agent 1: Triage and reformulate query
            logger.debug("Stage 1: Triage Specialist processing query")
            technical_query = await self.agent1.process(customer_query)
            logger.debug(f"Triage result: '{technical_query}'")
            
            # Agent 2: Search knowledge base and create draft
            logger.debug("Stage 2: Technical Expert searching knowledge base")
            draft_response = await self.agent2.process(technical_query)
            logger.debug(f"Technical Expert found {len(draft_response.sources)} sources")
            
            # Agent 3: Refine into customer-friendly response
            logger.debug("Stage 3: Communication Specialist refining response")
            final_answer = await self.agent3.process(
                customer_query, 
                draft_response.draft_solution, 
                draft_response.sources
            )
            
            result = {
                "final_answer": final_answer,
                "sources": draft_response.sources
            }
            
            logger.info(f"Successfully processed query with {len(draft_response.sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error in support system: {e}")
            return self._handle_error(customer_query, e)
    
    def _handle_error(self, customer_query: str, error: Exception) -> Dict[str, Any]:

        error_message = (
            "I apologize, but we're experiencing some technical difficulties at the moment. "
            "Please try again in a few minutes, or feel free to contact our support team directly "
            "if you need immediate assistance. We appreciate your patience!"
        )
        
        logger.error(f"Returning error response for query: '{customer_query}', Error: {error}")
        
        return {
            "final_answer": error_message,
            "sources": []
        }
    
    async def health_check(self) -> Dict[str, str]:

        return {
            "triage_specialist": "active",
            "technical_expert": "active", 
            "communication_specialist": "active"
        }


# Global support system instance
agent_system = ThreeAgentSupportSystem()