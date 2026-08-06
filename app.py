import os
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================


st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent ")

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(max_results=2)

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# ==========================================
# LLM
# ==========================================


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

# ==========================================
# PROMPT
# ==========================================

#prompt = hub.pull("hwchase17/react")
template = """
You are a helpful AI assistant.

You have access to the following tools:

{tools}

Tool Names:

{tool_names}

Use the following format:

Question: the input question

Thought: think carefully

If a tool is required:

Action: one of [{tool_names}]

Action Input: the input for the tool

Observation: the result

...

When you know the answer:

Final Answer: your answer

IMPORTANT:

- NEVER write Action: None
- NEVER write Action Input: None
- If the user simply greets you (Hi, Hello, Good Morning, Hey, etc.), DO NOT use any tool.
- Reply directly with:

Final Answer: Hello! 👋 How can I assist you today?

Question: {input}

Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)

# ==========================================
# UI INPUT
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")