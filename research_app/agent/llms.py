import os
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()


# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")



# llm_groq = ChatOpenRouter(
#     model="openrouter/free",
#     temperature=0.2,
#     max_tokens=None,
#     max_retries=2,
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )

groq_api_key = os.getenv("GROQ_API_KEY")

llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",  # or any Groq model
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key,
)