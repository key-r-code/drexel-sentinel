import os
import warnings
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.engine.tools import web_search, search_syllabi, add_to_calendar, delete_event

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# --- 1. SETUP ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "drexel-sentinel")
creds = service_account.Credentials.from_service_account_file(
    os.path.abspath("service_account.json")
).with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", credentials=creds, project=PROJECT_ID)

# --- 2. SPECIALIST AGENTS ---

advising_prompt = f"""You are an academic advisor. Use the search_syllabi tool to search through course documents for grading and policies. Be precise.

IMPORTANT: Today's date is {datetime.now().strftime('%B %d, %Y')} (YYYY-MM-DD: {datetime.now().strftime('%Y-%m-%d')}). Use this for any date calculations."""

calendar_agent_prompt = f"""
You are a high-level calendar assistant. 

CURRENT DATE: Today is {datetime.now().strftime('%B %d, %Y')} (YYYY-MM-DD: {datetime.now().strftime('%Y-%m-%d')}).

CRITICAL: When calling add_to_calendar tool:
- If time is unknown/TBD, pass an EMPTY STRING "" for time_str (NOT "TBD", NOT "None")
- If location is unknown/TBD, pass an EMPTY STRING "" for location (NOT "TBD", NOT "None")  
- If description/room is unknown/TBD, pass an EMPTY STRING "" for description (NOT "TBD", NOT "None")
- ALWAYS call the tool even if some fields are unknown - just use empty strings for those fields
- If users wants to modify an event, use the delete_event tool to delete the event first and then use the add_to_calendar tool to add the new event.

FORMATTING & ROOMS:
1. Event Title MUST be: '[COURSE]: [TYPE]' (e.g., 'MATH 291: Exam 1').
2. Put the room number (e.g., 'Bossone 201') into the 'description' field of the tool.
3. Use the formal building name (e.g., 'Bossone Research Enterprise Center') for the 'location' field.

DATE FORMATS:
- CRITICAL: When the user mentions dates like "February 4th" or "week 5", calculate the correct YEAR based on today's date ({datetime.now().strftime('%Y-%m-%d')}).
- Always use YYYY-MM-DD format for date_str parameter.
- If a date has already passed this academic year, it likely refers to next year.
"""

research_agent_prompt = f"""You are a researcher. Use the web_search tool to search for Drexel faculty news, research papers, and professor bios. You can also use the tool to look up general information asked.

IMPORTANT: Today's date is {datetime.now().strftime('%B %d, %Y')} (YYYY-MM-DD: {datetime.now().strftime('%Y-%m-%d')}). Use this for any date-related queries."""

memory = MemorySaver()

advising_agent = create_react_agent(
    model=llm, 
    tools=[search_syllabi], 
    checkpointer=memory, 
    prompt=advising_prompt
)

calendar_agent = create_react_agent(
    model=llm, 
    tools=[add_to_calendar, delete_event], 
    checkpointer=memory,  
    prompt=calendar_agent_prompt
)

research_agent = create_react_agent(
    model=llm, 
    tools=[web_search], 
    checkpointer=memory,  
    prompt=research_agent_prompt
)

# --- 3. Wrapper tools for the agents ---
# Store the base thread_id to create sub-agent thread IDs
_current_thread_id = None

def set_thread_id(thread_id: str):
    """Helper to set the current thread ID for sub-agents"""
    global _current_thread_id
    _current_thread_id = thread_id

@tool 
def calendar_assistant(request: str):
    """Add or remove calendar events. Use this tool when the user asks to add, remove, or modify calendar events.
    
    Input: Natural language scheduling request 
    """
    # Use a separate thread ID for the calendar agent to avoid message history conflicts
    sub_thread_id = f"{_current_thread_id}_calendar" if _current_thread_id else "calendar"
    config = {"configurable": {"thread_id": sub_thread_id}}
    result = calendar_agent.invoke({"messages": [("user", request)]}, config=config)
    return result["messages"][-1].content

@tool
def research_assistant(request: str):
    """Search the web for information. This tool is a wrapper for the web_search tool. Use this tool when the user asks to search the web for information.
    
    Input: Natural language search request (e.g., 'what is the weather in Philadelphia? or Tell me more about the professor of MATH 291')
    """
    # Use a separate thread ID for the research agent
    sub_thread_id = f"{_current_thread_id}_research" if _current_thread_id else "research"
    config = {"configurable": {"thread_id": sub_thread_id}}
    result = research_agent.invoke({"messages": [("user", request)]}, config=config)
    return result["messages"][-1].content

@tool
def advisor_assistant(request: str):
    """Search the course documents for information. This tool is a wrapper for the search_syllabi tool. Use this tool when the user asks to search the course documents for information.
    
    Input: Natural language search request (e.g., 'what is the grading policy for MATH 291?')
    """
    # Use a separate thread ID for the advising agent
    sub_thread_id = f"{_current_thread_id}_advisor" if _current_thread_id else "advisor"
    config = {"configurable": {"thread_id": sub_thread_id}}
    result = advising_agent.invoke({"messages": [("user", request)]}, config=config)
    return result["messages"][-1].content

# --- 4. Supervisor agent ---
SUPERVISOR_PROMPT = f"""
You are the Drexel Sentinel Executive Assistant. You are responsible for overseeing the work of the calendar_agent, research_agent, and advising_agent. You should also be able to answer general questions such as date and time, weather, and other general information.

CURRENT DATE: Today is {datetime.now().strftime('%B %d, %Y')} (YYYY-MM-DD: {datetime.now().strftime('%Y-%m-%d')}).

You will be given a user query and you will need to determine which agent to route the query to.
You will also need to determine if the user query is a scheduling request, a research request, or an advising request.
Infer the user's intent from the query and route the query to the appropriate agent. For example:
- If the user asks to add an event to the calendar, route to calendar_agent.
- If the user asks to search the web for information, route to research_agent.
- If the user asks to search the course documents for information, route to advisor_agent.
- If the user asks a general question, route to the appropriate agent based on the query.

Example:
User: "Add MATH 291 Exam 1 on Feb 4th"
You: Route to calendar_agent

User: "What is the weather in Philadelphia?"
You: Route to research_agent

User: "What is the grading policy for MATH 291?"
You: Route to advisor_agent
"""
supervisor_agent = create_react_agent(
    model=llm, 
    tools=[calendar_assistant, research_assistant, advisor_assistant], 
    checkpointer=memory,
    prompt=SUPERVISOR_PROMPT
)


def ask_sentinel(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    set_thread_id(thread_id)
    result = supervisor_agent.invoke({"messages": [("user", user_input)]}, config=config)
    return result["messages"][-1].content