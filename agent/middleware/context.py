import json
from typing import Any
from langchain_core.messages import ToolMessage


def json_payload(message:ToolMessage)->Any:

    if not isinstance(message.content,str):
        return message.content
    try:
        return json.loads(message.content)
    except json.JSONDecodeError:
        return None

#Use this as a node after Internet tool_call and before the tool_call content reaches llm        
def internet_tool_context(payload:dict)->dict|None:
    """Extracts only useful Internet tool context """
    
    output = payload.get("content")
    
    if not isinstance(output,dict):
        return None
    context = {
        "title": output.get("title"),
        "content": output.get("content"),
        "url": output.get("url"),
    }
    context = {key: value for key, value in context.items() if value}
    return context or None