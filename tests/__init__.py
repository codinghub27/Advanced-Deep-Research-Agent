"""
Test package. The environment below is forced BEFORE research_app is imported so
tests never use real credentials, never touch a real database and never send
LangSmith traces. (load_dotenv() does not override variables that already exist.)
"""
import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["SECRET_KEY"] = "test-secret-key-not-real"
os.environ["GROQ_API_KEY"] = "test-groq-key-not-real"
os.environ["TAVILY_API_KEY"] = "tvly-test-key-not-real"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("ADMIN_USERNAMES", None)
