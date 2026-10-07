from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.config.state import State
from agent.nodes.typesafe_routing import _Orchestrator_router
from agent.subgraphs.chatbot import ChatbotAgent
from agent.subgraphs.gmail import GmailAgent
from agent.subgraphs.internet import InternetAgent
from agent.subgraphs.local import LocalAgent

load_dotenv(override=True)


class Agent:
    def __init__(self):
        self.orchestrator = None
        self.memory = None
        self.graph = None
        self.handoff_tools = None
        self.checkpointer = None
        self.store = None
        self.internet_agent = None
        self.gmail_agent = None
        self.local_agent = None
        self.chat_bot = None

    async def setup(self, state: State):
        self.checkpointer = InMemorySaver()
        
        self.orchestrator = _Orchestrator_router

        self.internet_agent = InternetAgent()
        await self.internet_agent.build_graph(state)

        self.gmail_agent = GmailAgent()
        await self.gmail_agent.build_graph(state)

        self.local_agent = LocalAgent()
        await self.local_agent.build_graph(state)

        self.chat_bot = ChatbotAgent()
        await self.chat_bot.build_graph(state)

        self.build_graph(state)

    def build_graph(self, state: State):
        graph_builder = StateGraph(State)

        graph_builder.add_node("orchestrator", self.orchestrator)
        graph_builder.add_node("internet", self.internet_agent.graph)
        graph_builder.add_node("gmail", self.gmail_agent.graph)
        graph_builder.add_node("local", self.local_agent.graph)
        graph_builder.add_node("chatbot", self.chat_bot.graph)

        graph_builder.add_edge(START, "orchestrator")
        graph_builder.add_conditional_edges(
            "orchestrator",
            lambda state: state["next"],
            {
                "Internet": "internet",
                "Gmail": "gmail",
                "worker": "local",
                "assistant": "chatbot",
            },
        )
        graph_builder.add_edge("internet", END)
        graph_builder.add_edge("gmail", END)
        graph_builder.add_edge("local", END)
        graph_builder.add_edge("chatbot", END)

        self.graph = graph_builder.compile(checkpointer=self.checkpointer)
