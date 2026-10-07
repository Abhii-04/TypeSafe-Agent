import os
from dotenv import load_dotenv
from mem0 import Memory

memory_config = {
    "llm": {
        "provider":"deepseek",
        "config":{
            "api_key":os.getenv('DEEPSEEK_API_KEY'),
            "model":"deepseek-v4-flash",
            "temperature":0.2
        },
    },

    #local embedding model, so no OpenAI embedding key will be needed
    "embedded":{
        "provider":"huggingface",
        "config":{
            "model":"multi-qa-MiniLM-L-cos-v1",
        },
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "collection_name": "memories",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 384,
        },
    },
}

memory = Memory.from_config(memory_config)