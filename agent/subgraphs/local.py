"""You are a local machine and workflow assistant."""

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


class LocalAgent:
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

    def LocalAgent(self, state: State):
        system_message = f"""You handle offline tasks on the user's local operating system.

Task instructions from the orchestrator:
{state.get("task_instructions") or "Handle the latest user request."}

Your work area and boundaries:
- Manage installed applications, windows, local files, folders, and operating-system settings using available local tools.
- Work only on the local machine. Do not browse the web, use internet services, access Gmail, or download software.
- Use available local tools when a task requires access to the machine.
- Do not claim to read files, run commands, or change the machine without tool results.
- Ask for confirmation before destructive actions.
- Leave web research and Gmail tasks to their respective specialists.
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

    def Local_router(self, state: State):
        message = state["messages"][-1]
        if self.tools and isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            return "tools"
        return END

    async def build_graph(self, state: State):
        """Build the specialist subgraph."""
        self.setup(state)
        graph_builder = StateGraph(State)

        graph_builder.add_node("Local_agent", self.LocalAgent)
        if self.tools:
            graph_builder.add_node("tools", ToolNode(self.tools))

        graph_builder.add_edge(START, "Local_agent")
        graph_builder.add_conditional_edges(
            "Local_agent",
            self.Local_router,
            {"tools": "tools", END: END} if self.tools else {END: END},
        )
        if self.tools:
            graph_builder.add_edge("tools", "Local_agent")

        self.graph = graph_builder.compile()

    async def run_superstep(self, message, history):
        config = {"configurable": {"thread_id": "local"}}
        state = {"messages": [HumanMessage(content=message)]}
        result = await self.graph.ainvoke(state, config=config)
        print(result["messages"][-1].content)
        return result
