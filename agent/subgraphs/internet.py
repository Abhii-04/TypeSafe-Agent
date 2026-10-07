"""This agent works/peforms all tasks over the internet and does not peform any tasks on local machine. """

from typing import Annotated,Any,TypedDict,List
from datetime import datetime,timedelta
import os

from langchain_core.caches import InMemoryCache
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver

from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode


from agent.config.state import State
def llm():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key :
        raise ValueError("api key not found")
    return ChatOpenAI(
        api_key = api_key,
        model = 'deepseek-v4-flash',
        base_url="https://api.deepseek.com"
    )

class InternetAgent:
    def __init__(self):
        self.agent=None
        self.memory=None
        self.graph=None
        self.tools=None

    def setup(self,state:State):
        """ Agent setup """
        tools=[]
        self.tools = tools
        self.agent = llm()
        if self.tools:
            self.agent = self.agent.bind_tools(self.tools)
        self.memory = InMemorySaver()

    def InternetAgent(self,state:State):
        system_message=f"""ou are an internet research agent.

Task instructions from the orchestrator:
{state.get("task_instructions") or "Handle the latest user request."}

Your work area:
- Use the Internet_search tool for current, recent, external, or source-backed information.
- Search when facts may have changed or when the user asks for latest/current information.
- Use browser tools when the user asks you to open, inspect, navigate, or interact with a specific website.
- Use browser_snapshot before clicking or filling so you can get valid element refs.
- Summarize findings clearly and mention source names or URLs when the tool result provides them.
- If search results conflict, say so and prefer more authoritative or recent sources.

Boundaries:
- Browser actions can change page state. Ask for confirmation before form submission, purchases, applications, account changes, destructive actions, or sending messages.
- Do not claim to run code or read local files.
- Do not handle Gmail or email tasks. Say that the Gmail workflow should handle them.
- Do not handle LinkedIn MCP tasks. Say that the LinkedIn workflow should handle them.
- If the task does not need internet research, answer briefly that it should be handled by the general assistant.
- If the available search results are insufficient, say what is missing instead of guessing.
- Do not describe tool calls or tool syntax in your response text. If you need a tool, use the bound tool-calling interface only.

Skills:
- You have access to read_skill, which reads local instructions from skills/<skill>/SKILL.md.
- When the user names a skill or the task clearly matches one, call read_skill before researching or answering and follow those instructions.
- If read_skill reports a missing, invalid, or empty skill, create a concise report in your response with: Error, Cause, Solution, and Next step.
- Do not invent skill instructions when the skill file cannot be read.

Loaded skill instructions:
{state.get("internet_skill", "")}

The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. """

        if state.get("feedback_on_work"):
            system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state["feedback_on_work"]}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user. """
        
        
        messages = [SystemMessage(content=system_message)] + state["messages"]
        response = self.agent.invoke(messages)
        return {"messages": [response]}

    def Internet_router(self,state:State):
        message= state["messages"][-1]
        if isinstance(message,AIMessage) and getattr(message,"tool_calls",None):
            return "tools"
        return END
    
    async def build_graph(self,state:State):
        """graph building node"""
        self.setup(state)
        graph_builder = StateGraph(State)

        #Nodes
        graph_builder.add_node("Internet_agent",self.InternetAgent)
        graph_builder.add_node("tools",ToolNode(self.tools))

        #Edges
        graph_builder.add_edge(START,"Internet_agent")
        graph_builder.add_conditional_edges(
            "Internet_agent",
            self.Internet_router,
            {
                "tools":"tools",
                END:END
            }
        )
        graph_builder.add_edge("tools","Internet_agent")

        self.graph = graph_builder.compile()
    
    async def run_superstep(self,message,history):
        config = {"configurable": {"thread_id": self.agent_id}}
        state = {"messages": [HumanMessage(content=message)]}
        result = await self.graph.ainvoke(state, config=config)
        print(result["messages"][-1].content)
        return result
