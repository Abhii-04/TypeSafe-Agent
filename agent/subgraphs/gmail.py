"""You are a Gmail assistant."""

from datetime import datetime
import os

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config.state import State


def llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("api key not found")
    return ChatOpenAI(
        api_key=api_key,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )


class GmailAgent:
    def __init__(self):
        self.agent = None
        self.memory = None
        self.graph = None
        self.tools = None

    def setup(self, state: State):
        """Agent setup."""
        tools = []
        self.tools = tools
        self.agent = llm()
        if self.tools:
            self.agent = self.agent.bind_tools(self.tools)
        self.memory = InMemorySaver()

    def GmailAgent(self, state: State):
        system_message = f"""You are a Gmail assistant.

Task instructions from the orchestrator:
{state.get("task_instructions") or "Handle the latest user request."}

Your work area and boundaries:
- Help with email requests, drafts, inbox questions, and Gmail workflows.
- Use available Gmail tools when an action requires account access.
- Do not claim to read, send, or change emails without a successful tool result.
- Ask for confirmation before sending emails or performing destructive actions.
- If Gmail tools are unavailable, explain what is missing.
- Do not describe tool syntax in your response. Use the bound tool-calling interface.

The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}."""

        if state.get("feedback_on_work"):
            system_message += f"""
Your previous reply did not meet the success criteria.
Feedback:
{state["feedback_on_work"]}
Continue the assignment using this feedback or ask the user a question."""

        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = self.agent.invoke(messages)
        return {"messages": [response]}

    def Gmail_router(self, state: State):
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            return "tools"
        return END

    async def build_graph(self, state: State):
        """Build the specialist subgraph."""
        self.setup(state)
        graph_builder = StateGraph(State)

        graph_builder.add_node("Gmail_agent", self.GmailAgent)
        graph_builder.add_node("tools", ToolNode(self.tools))

        graph_builder.add_edge(START, "Gmail_agent")
        graph_builder.add_conditional_edges(
            "Gmail_agent",
            self.Gmail_router,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "Gmail_agent")

        self.graph = graph_builder.compile()

    async def run_superstep(self, message, history):
        config = {"configurable": {"thread_id": "gmail"}}
        state = {"messages": [HumanMessage(content=message)]}
        result = await self.graph.ainvoke(state, config=config)
        print(result["messages"][-1].content)
        return result
