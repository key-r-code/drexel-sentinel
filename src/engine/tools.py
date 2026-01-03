import os
import sqlite3
import datetime
import logging
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from googleapiclient.discovery import build
from google.oauth2 import service_account

load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calendar_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
CRED_PATH = os.path.abspath("service_account.json")
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
TIMEZONE = "America/New_York"

logger.info(f"Calendar ID configured: {CALENDAR_ID}")

# --- AUTH ---
SCOPES = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/calendar']
creds = service_account.Credentials.from_service_account_file(CRED_PATH).with_scopes(SCOPES)
calendar_service = build('calendar', 'v3', credentials=creds)

# --- WEB SEARCH TOOL ---
web_search = TavilySearch(max_results=3)

# --- EXISTING TOOLS ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", credentials=creds)
vector_db = Chroma(persist_directory="db", embedding_function=embeddings)

@tool
def search_syllabi(query: str):
    """Search course syllabuses for academic policies, grading, and office hours."""
    logger.info(f"search_syllabi called with query: {query}")
    docs = vector_db.similarity_search(query, k=3)
    result = "\n\n".join([d.page_content for d in docs])
    logger.info(f"search_syllabi returned {len(docs)} documents")
    return result

@tool
def add_to_calendar(title: str, date_str: str, time_str: str = "", location: str = "", description: str = ""):
    """
    Adds a detailed event to the Google Calendar.
    - title: [COURSE]: [TYPE] format (e.g., MATH 291: Exam 1)
    - date_str: YYYY-MM-DD
    - time_str: HH:MM (24-hour format). Leave empty string if time is unknown or TBD.
    - location: Building name. Leave empty string if unknown or TBD.
    - description: Room number or additional notes. Leave empty string if unknown or TBD.
    """
    
    logger.info("="*60)
    logger.info("ADD_TO_CALENDAR FUNCTION CALLED")
    logger.info(f"Title: {title}")
    logger.info(f"Date: {date_str}")
    logger.info(f"Time: {time_str}")
    logger.info(f"Location: {location}")
    logger.info(f"Description: {description}")
    logger.info(f"Calendar ID: {CALENDAR_ID}")
    logger.info("="*60)
    
    # Graphite (Color ID 8) for exams
    color_id = "8" if any(kw in title.lower() for kw in ["exam", "midterm", "quiz"]) else None
    
    # Check if time is provided and valid (not empty, not "TBD", not "None")
    is_timed = (
        time_str 
        and time_str.strip() 
        and time_str.upper() not in ["TBD", "NONE", "UNKNOWN", "N/A"]
    )
    
    event = None  # Initialize event
    
    if is_timed:
        try:
            start_time = f"{date_str}T{time_str}:00"
            start_obj = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            end_time = (start_obj + datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
            
            event = {
                'summary': title,
                'location': location if location and location.upper() not in ["TBD", "NONE", "UNKNOWN"] else None,
                'description': description if description and description.upper() not in ["TBD", "NONE", "UNKNOWN"] else None,
                'colorId': color_id,
                'start': {'dateTime': start_time, 'timeZone': TIMEZONE},
                'end': {'dateTime': end_time, 'timeZone': TIMEZONE},
            }
            logger.info(f"Created timed event object")
        except ValueError as e:
            # If time parsing fails, fall through to create all-day event
            logger.warning(f"Time parsing failed: {e}, creating all-day event instead")
            event = None
            
    # Create all-day event if not timed or if timed parsing failed
    if event is None:
        event = {
            'summary': title,
            'location': location if location and location.strip() and location.upper() not in ["TBD", "NONE", "UNKNOWN"] else None,
            'description': description if description and description.strip() and description.upper() not in ["TBD", "NONE", "UNKNOWN"] else None,
            'colorId': color_id,
            'start': {'date': date_str},
            'end': {'date': date_str},
        }
        logger.info(f"Created all-day event object")

    logger.info(f"Event object: {event}")
    
    try:
        result = calendar_service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        logger.info(f"Google API Response: {result}")
        logger.info(f"Event created successfully with ID: {result.get('id')}")
        return f"Successfully added '{title}' for {date_str} (Room: {description or 'N/A'})."
    except Exception as e:
        logger.error(f"ERROR adding event: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error adding event: {str(e)}"

@tool
def delete_event(event_title: str, date_str: str):
    """Deletes an event from the calendar."""
    logger.info("="*60)
    logger.info("DELETE_EVENT FUNCTION CALLED")
    logger.info(f"Event title: {event_title}")
    logger.info(f"Date: {date_str}")
    logger.info(f"Calendar ID: {CALENDAR_ID}")
    logger.info("="*60)
    
    try:
        t_min, t_max = f"{date_str}T00:00:00Z", f"{date_str}T23:59:59Z"
        events = calendar_service.events().list(
            calendarId=CALENDAR_ID, 
            timeMin=t_min, 
            timeMax=t_max, 
            q=event_title
        ).execute().get('items', [])
        
        logger.info(f"Found {len(events)} matching event(s)")
        for event in events:
            logger.info(f"  - {event.get('summary')} (ID: {event.get('id')})")
        
        if not events:
            logger.warning(f"No events found matching '{event_title}' on {date_str}")
            return f"No events found matching '{event_title}' on {date_str}."
        
        for event in events:
            calendar_service.events().delete(calendarId=CALENDAR_ID, eventId=event['id']).execute()
            logger.info(f"Deleted event: {event.get('summary')} (ID: {event.get('id')})")
        
        return f"Removed {len(events)} event(s) matching '{event_title}' from {date_str}."
    except Exception as e:
        logger.error(f"ERROR deleting event: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error deleting event: {str(e)}"

tools = [search_syllabi, add_to_calendar, delete_event, web_search]