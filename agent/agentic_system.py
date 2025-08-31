import logging
from typing import Dict, Any, List, TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from configs.config import config
from agent.agent_tools import SUPPORT_TOOLS

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: List[Any]
    original_query: str
    technical_query: Optional[str]
    sources: List[str]
    draft_solution: Optional[str]
    final_answer: Optional[str]
    current_step: str
    error: Optional[str]
    metadata: Dict[str, Any]


class AgenticSupportSystem:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        ) if config.OPENAI_API_KEY else None

        # ToolNode executes tools in workflow automatically
        self.tool_node = ToolNode(SUPPORT_TOOLS)

        # Workflow
        self.workflow = self._create_workflow()

        # Memory
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        logger.info("Agentic Support System initialized with LangGraph workflow")

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("coordinator", self._coordinator_agent)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("finalizer", self._finalizer_agent)

        workflow.set_entry_point("coordinator")

        workflow.add_conditional_edges(
            "coordinator",
            lambda s: s["current_step"] in {"triaging", "searching", "drafting", "finalizing"},
            {"tools": "tools", END: "finalizer"}
        )

        # Coordinator decides next node; no conditional boolean edges needed
        workflow.add_edge("tools", "coordinator")
        workflow.add_edge("finalizer", END)

        return workflow

    def _coordinator_agent(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content="Coordinator agent for customer support."))

        step = state["current_step"]

        # Determine next tool
        if step == "start":
            state["current_step"] = "triaging"
            state["tool_to_run"] = "triage_query"
        elif step == "triaging" and state.get("technical_query") is None:
            state["tool_to_run"] = "triage_query"
        elif step == "triaging" and state.get("technical_query"):
            state["current_step"] = "searching"
            state["tool_to_run"] = "search_knowledge_base"
        elif step == "searching" and not state.get("sources"):
            state["tool_to_run"] = "search_knowledge_base"
        elif step == "searching" and state.get("sources"):
            state["current_step"] = "drafting"
            state["tool_to_run"] = "generate_draft_solution"
        elif step == "drafting" and not state.get("draft_solution"):
            state["tool_to_run"] = "generate_draft_solution"
        elif step == "drafting" and state.get("draft_solution"):
            state["current_step"] = "finalizing"
            state["tool_to_run"] = "generate_customer_response"
        elif step == "finalizing" and not state.get("final_answer"):
            state["tool_to_run"] = "generate_customer_response"
        elif step == "finalizing" and state.get("final_answer"):
            state["current_step"] = "complete"
            state.pop("tool_to_run", None)

        state["messages"] = messages
        logger.info(f"Coordinator reached step '{state['current_step']}'")
        return state


    def _update_state_from_messages(self, state: AgentState) -> AgentState:
        """Update AgentState from recent ToolMessages."""
        try:
            messages = state.get("messages", [])
            for message in reversed(messages[-20:]):
                if isinstance(message, ToolMessage):
                    for prev in reversed(messages):
                        if getattr(prev, "tool_calls", None):
                            for call in prev.tool_calls:
                                if call.get("id") == getattr(message, "tool_call_id", None):
                                    name = call["name"]
                                    content = message.content

                                    if name == "triage_query":
                                        state["technical_query"] = content.strip('"')
                                    elif name == "search_knowledge_base":
                                        import json
                                        try:
                                            parsed = json.loads(content) if content.startswith("{") else {}
                                            state["sources"] = parsed.get("sources", [])
                                        except Exception:
                                            state["sources"] = []
                                    elif name == "generate_draft_solution":
                                        state["draft_solution"] = content
                                    elif name == "generate_customer_response":
                                        state["final_answer"] = content
                                    break
                            break
            return state
        except Exception as e:
            logger.error(f"Error updating state from messages: {e}")
            return state

    def _finalizer_agent(self, state: AgentState) -> AgentState:
        state = self._update_state_from_messages(state)

        if state.get("error"):
            state["final_answer"] = (
                "I apologize, but we're experiencing technical difficulties. "
                "Please try again later or contact support."
            )
            state["sources"] = []
        elif not state.get("final_answer"):
            state["final_answer"] = (
                "I couldn't find specific information for your query. "
                "Please contact our support team for assistance."
            )

        logger.info("Finalizer completed workflow processing")
        return state

    async def process_support_query(self, customer_query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing agentic support query: '{customer_query}'")

            initial_state = AgentState(
                messages=[],
                original_query=customer_query,
                technical_query=None,
                sources=[],
                draft_solution=None,
                final_answer=None,
                current_step="start",
                error=None,
                metadata={}
            )

            thread_id = f"support_{hash(customer_query)}_{id(self)}"
            final_state = None

            for state in self.app.stream(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values"
            ):
                final_state = state

            if not final_state:
                raise Exception("Workflow did not produce a final state")

            final_state = self._update_state_from_messages(final_state)

            result = {
                "final_answer": final_state.get("final_answer", "No response generated"),
                "sources": final_state.get("sources", [])
            }

            logger.info(f"Agentic workflow completed with {len(result['sources'])} sources")
            return result

        except Exception as e:
            logger.error(f"Error in agentic support system: {e}")
            return {
                "final_answer": (
                    "I apologize, but we're experiencing technical difficulties. "
                    "Please try again later or contact support."
                ),
                "sources": []
            }

    async def health_check(self) -> Dict[str, str]:
        try:
            if self.app and self.workflow:
                return {
                    "coordinator": "active",
                    "tools": "active", 
                    "finalizer": "active",
                    "workflow": "compiled",
                    "tools_available": str(len(SUPPORT_TOOLS))
                }
            else:
                return {
                    "coordinator": "error",
                    "tools": "error",
                    "finalizer": "error",
                    "workflow": "not_compiled"
                }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "coordinator": "error",
                "tools": "error",
                "finalizer": "error",
                "workflow": "error"
            }


# Instantiate
agentic_system = AgenticSupportSystem()
