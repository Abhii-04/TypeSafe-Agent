"""You are a general conversational assistant."""

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


class ChatbotAgent:
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

    def ChatbotAgent(self, state: State):
        system_message = f"""You are a companion for casual chatting only.

Task instructions from the orchestrator:
{state.get("task_instructions") or "Chat casually with the user."}

Your work area and boundaries:
- Respond naturally to greetings, small talk, everyday conversation, and the user's thoughts or feelings.
- Use the relevant conversation history to understand follow-up requests.
- Do not perform tasks, research, coding, writing assignments, email operations, or local machine actions.
- If asked to perform a task, briefly explain that the appropriate specialist should handle it.
- Do not call tools or claim to perform actions.

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

    def Chatbot_router(self, state: State):
        return END

    async def build_graph(self, state: State):
        """Build the specialist subgraph."""
        self.setup(state)
        graph_builder = StateGraph(State)

        graph_builder.add_node("Chatbot_agent", self.ChatbotAgent)

        graph_builder.add_edge(START, "Chatbot_agent")
        graph_builder.add_conditional_edges(
            "Chatbot_agent",
            self.Chatbot_router,
            {END: END},
        )

        self.graph = graph_builder.compile()

    async def run_superstep(self, message, history):
        config = {"configurable": {"thread_id": "chatbot"}}
        state = {"messages": [HumanMessage(content=message)]}
        result = await self.graph.ainvoke(state, config=config)
        print(result["messages"][-1].content)
        return result
