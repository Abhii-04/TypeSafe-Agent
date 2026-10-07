from langchain_core.messages import AIMessage
from langgraph.graph.state import RunnableConfig
from langgraph.types import interrupt
from typing import Callable
from agent.config.state import State
from langchain_core.tools import tool,BaseTool
from langchain_core.messages import ToolMessage

RISKY_TOOLS=["bash","write_file"]

def add_approval(main_tool:Callable|BaseTool)->BaseTool:
    """"wrap a tool to support human approval for risky tools """
    if not isinstance(main_tool,BaseTool):
        main_tool = tool(main_tool)

    @tool(
        main_tool.name,
        description = main_tool.description,
        args_schema=main_tool.args_schema
    )
    def call_main_tool_with_HITL(config:RunnableConfig,**tool_input):
        descision = interrupt({
            "awaiting":main_tool.name,
            "args":tool_input,
        })

        #tools approved
        if isinstance(descision,dict) and descision.get("approved"):
            return main_tool.invoke(tool_input.config)

        #tools rejected
        return "Cancelled by human, Continue without executing this tool and provide next stage."
    return call_main_tool_with_HITL

#use this HITL for security purpose as it will always block risky tools and ask for human approval.
def halt_on_risky_tools(state:State):
    """human in the loop to prevent misues of risky tools."""
    last = state["messages"][-1]
    if isinstance(last,AIMessage) and getattr(last,"tool_calls",None):
        risky_tool_calls = [
            tc
            for tc in last.tool_calls
            if tc.get("name") in RISKY_TOOLS
        ]

        for tc in risky_tool_calls:
            descision=interrupt({"awaiting":tc["name"],"args":tc.get("args",{})})

            if not isinstance(descision,dict) or not descision.get("approved"):
                tool_messages = [
                    ToolMessage(
                        content="Cancelled by human. Continue without executing this tool and provide next steps.",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                    for tool_call in last.tool_calls
                ]
                return {"messages":tool_messages,"hitl_decision":"rejected"}

        if risky_tool_calls:
            return {"hitl_decision":"approved"}

    return {"hitl_decision":"approved"}


#Use this tool to ask human a question/clarification
@tool
def ask_question(question:str):
    """Ask the Human one question using this tool"""
    response = interrupt({question:question})
    return{
        "users response":response
    }