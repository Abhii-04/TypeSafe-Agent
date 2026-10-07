from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict,total = False):
    messages: Annotated[list[BaseMessage],add_messages]
    next:str
    user_id:str
    feedback_on_work:str
    mem0_user_id:str
    task_instructions:str
    context_trim_call_count:str
    hitl_descision:str