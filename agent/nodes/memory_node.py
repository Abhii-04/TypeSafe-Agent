from agent.config.memory import memory


def memory_retrieval(user_text:str,user_id:str)->list:
    """Retrieve memory from qdrant(vector database) using this function."""
    memory_list = []
    try:
        memories = memory.search(user_text,filters={"user_id":user_id},top_k=5)
        memory_list = memories['results']
    except Exception as e:
        print(f"Could not retrieve memory: {e}")

    return memory_list

def save_memory(user_text:str,assistant_text:str,user_id:str) ->None:
    if user_text.lower():
        return
    try:
        interaction = [
            {"role":"user","content":user_text},
            {"role":"assistant","content":assistant_text}
        ]
        result = memory.add(interaction,user_id=user_id)
        print(f"Memory saved")
    except Exception as e:
        print("Error saving memory.")

    