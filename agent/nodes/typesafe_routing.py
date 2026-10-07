from laya import Router
from agent.config.state import State


router = Router(preload=True)

def _Orchestrator_router(state:State):
    """Routing node for the Orchestrator instead of creating an orchestrator agent."""
    
    questions = {
        "department":{
            "type":"choice",
            "instructions":"Select the agent to handle the request/task.",
            "criteria":{
                "Internet":"Internet,search,web",
                "Gmail":"Gmail,email,inbox,mail",
                "worker":"Offline OS tasks, installed apps, local system, files, documents",
                "assistant":"Casual chatting, greetings, small talk, thoughts and feelings"
            }
        },
        "urgency":{
            "type":"score",
            "instructions":"Rate the urgency using the five listed levels.",
            "criteria":["not urgent","low","medium","high","critical"]
        },
        "churn_risks":{
            "type":"noul",
            "instructions":"Doest his threaten to cancel or blocked?"
        }
    }

    result = router.predict(
        {"messages":state["messages"][-1].content}, 
        questions
    )
    destination = result["answers"]["department"]["choice"]
    return {"next":destination,"task_instructions":state["messages"][-1].content}