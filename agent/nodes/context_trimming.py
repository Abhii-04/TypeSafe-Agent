from typing import Any
from langchain_core.messages.utils import trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES,RemoveMessage
from agent.config.state import State

TRIM_EVERY_CALLS = 2
MAX_CONTEXT_TOKENS=2000

def context_trimming_node(state:State) ->dict[str,Any]:
    """trim context of Agent after every 2nd llm call."""
    messages = state["messages"][-1]
    call_count = state.get("context_trim_call_count",0)+1

    if call_count < TRIM_EVERY_CALLS:
        return {"context_trim_call_count":call_count}

    trimmed_messages = trim_messages(
        messages,
        max_tokens = MAX_CONTEXT_TOKENS,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )

    if len(trimmed_messages)==len(messages):
        return {"context_trim_call_counter":0}

    return{
        "context_trim_call_counter":0,
        "messages":[
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed_messages,
        ],
    }