from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import json
import redis
import hashlib
from datetime import datetime, timezone
import uuid
from file_storage import FileStorage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="HighTribe Chatbot", version="1.0.0")

# Initialize Redis client
redis_client = None
file_storage = None

try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    # Test Redis connection
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.warning(f"Redis not available: {str(e)}. Using file-based persistent storage.")
    redis_client = None
    file_storage = FileStorage()

# In-memory fallback storage when both Redis and file storage fail
in_memory_storage = {}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    gmail: EmailStr
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

class ChatHistoryItem(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class ChatSession(BaseModel):
    session_id: str
    gmail: str
    messages: List[ChatHistoryItem]
    created_at: str
    updated_at: str

# Helper functions for Redis operations
def get_user_key(gmail: str) -> str:
    """Generate a unique Redis key for user's chat history"""
    # Hash the email for privacy and consistent key generation
    email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
    return f"chat_history:{email_hash}"

def get_session_key(gmail: str, session_id: str) -> str:
    """Generate a Redis key for a specific chat session"""
    email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
    return f"chat_session:{email_hash}:{session_id}"

def save_message_to_redis(gmail: str, session_id: str, role: str, content: str):
    """Save a chat message to Redis, file storage, or in-memory storage"""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    message_data = {
        "role": role,
        "content": content,
        "timestamp": timestamp
    }
    
    if redis_client:
        try:
            session_key = get_session_key(gmail, session_id)
            user_key = get_user_key(gmail)
            
            # Save message to session
            redis_client.rpush(session_key, json.dumps(message_data))
            
            # Set expiration for session (30 days)
            redis_client.expire(session_key, 30 * 24 * 60 * 60)
            
            # Update user's session list
            session_info = {
                "session_id": session_id,
                "last_updated": timestamp
            }
            redis_client.hset(user_key, session_id, json.dumps(session_info))
            redis_client.expire(user_key, 30 * 24 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Error saving message to Redis: {str(e)}")
    elif file_storage:
        # Use file-based persistent storage
        file_storage.save_message(gmail, session_id, role, content)
    else:
        # Use in-memory storage as last resort
        email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
        
        if email_hash not in in_memory_storage:
            in_memory_storage[email_hash] = {}
        
        if session_id not in in_memory_storage[email_hash]:
            in_memory_storage[email_hash][session_id] = {
                "messages": [],
                "last_updated": timestamp
            }
        
        in_memory_storage[email_hash][session_id]["messages"].append(message_data)
        in_memory_storage[email_hash][session_id]["last_updated"] = timestamp

def get_chat_history(gmail: str, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve chat history for a specific session"""
    if redis_client:
        try:
            session_key = get_session_key(gmail, session_id)
            messages = redis_client.lrange(session_key, -limit, -1)  # Get last 'limit' messages
            
            chat_history = []
            for message_json in messages:
                try:
                    message_data = json.loads(message_json)
                    chat_history.append(message_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message: {message_json}")
                    continue
                    
            return chat_history
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    elif file_storage:
        # Use file-based persistent storage
        return file_storage.get_chat_history(gmail, session_id, limit)
    else:
        # Use in-memory storage
        email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
        
        if email_hash in in_memory_storage and session_id in in_memory_storage[email_hash]:
            messages = in_memory_storage[email_hash][session_id]["messages"]
            return messages[-limit:] if limit else messages
        
        return []

def get_user_sessions(gmail: str) -> List[Dict[str, Any]]:
    """Get all sessions for a user"""
    if redis_client:
        try:
            user_key = get_user_key(gmail)
            sessions_data = redis_client.hgetall(user_key)
            
            sessions = []
            for session_id, session_info_json in sessions_data.items():
                try:
                    session_info = json.loads(session_info_json)
                    session_info['session_id'] = session_id
                    sessions.append(session_info)
                except json.JSONDecodeError:
                    continue
            
            # Sort by last updated (most recent first)
            sessions.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving user sessions: {str(e)}")
            return []
    elif file_storage:
        # Use file-based persistent storage
        return file_storage.get_user_sessions(gmail)
    else:
        # Use in-memory storage
        email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
        
        if email_hash in in_memory_storage:
            sessions = []
            for session_id, session_data in in_memory_storage[email_hash].items():
                sessions.append({
                    "session_id": session_id,
                    "last_updated": session_data["last_updated"]
                })
            
            # Sort by last updated (most recent first)
            sessions.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
            return sessions
        
        return []

def create_openai_messages(chat_history: List[Dict[str, Any]], current_message: str) -> List[Dict[str, str]]:
    """Convert chat history to OpenAI messages format"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant for HighTribe, a travel planning service focused on Pakistan. Provide helpful, concise, and friendly responses about travel, destinations, and general assistance. Use the conversation history to provide contextual and personalized responses."}
    ]
    
    # Add historical messages
    for msg in chat_history:
        if msg['role'] in ['user', 'assistant']:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    return messages

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chatbot page"""
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    return templates.TemplateResponse(
        "chatbot.html", 
        {
            "request": request, 
            "google_maps_api_key": google_maps_api_key
        }
    )

@app.post("/api/chat")
async def chat_with_bot_stream(chat_message: ChatMessage):
    """Handle chat messages with ChatGPT integration, streaming response, and chat history"""
    try:
        logger.info(f"Received chat message from {chat_message.gmail}: {chat_message.message}")
        message = chat_message.message.strip()
        gmail = chat_message.gmail
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Get chat history for context
        chat_history = get_chat_history(gmail, session_id, limit=10)  # Last 10 messages for context
        
        # Save user message
        save_message_to_redis(gmail, session_id, "user", message)
        
        def generate_response():
            try:
                # Prepare messages with history context
                openai_messages = create_openai_messages(chat_history, message)
                
                logger.info(f"Calling OpenAI API with {len(openai_messages)} messages (including history)...")
                
                # Call OpenAI API with streaming
                stream = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=openai_messages,
                    max_tokens=2000,
                    temperature=0.7,
                    stream=True
                )
                
                # Collect the complete response to save to history
                complete_response = ""
                
                # Send session ID first
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        complete_response += content
                        # Send each chunk as Server-Sent Events format
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Save assistant response to history
                if complete_response.strip():
                    save_message_to_redis(gmail, session_id, "assistant", complete_response)
                
                # Send end signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}")
                yield f"data: {json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'Sorry, I encountered an error. Please try again.'})}\n\n"]),
            media_type="text/plain"
        )

@app.get("/api/chat/history/{gmail}")
async def get_user_chat_history(gmail: str):
    """Get all chat sessions for a user"""
    try:
        sessions = get_user_sessions(gmail)
        return {
            "gmail": gmail,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/api/chat/session/{gmail}/{session_id}")
async def get_session_messages(gmail: str, session_id: str, limit: int = 50):
    """Get messages for a specific chat session"""
    try:
        messages = get_chat_history(gmail, session_id, limit)
        return {
            "gmail": gmail,
            "session_id": session_id,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error retrieving session messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session messages")

@app.delete("/api/chat/session/{gmail}/{session_id}")
async def delete_chat_session(gmail: str, session_id: str):
    """Delete a specific chat session"""
    try:
        if redis_client:
            session_key = get_session_key(gmail, session_id)
            user_key = get_user_key(gmail)
            
            # Delete session messages
            redis_client.delete(session_key)
            
            # Remove session from user's session list
            redis_client.hdel(user_key, session_id)
            
            return {"message": "Session deleted successfully"}
        elif file_storage:
            # Use file-based storage
            success = file_storage.delete_session(gmail, session_id)
            if success:
                return {"message": "Session deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Use in-memory storage
            email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
            
            if email_hash in in_memory_storage and session_id in in_memory_storage[email_hash]:
                del in_memory_storage[email_hash][session_id]
                return {"message": "Session deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "disconnected"
    storage_type = "in-memory"
    storage_stats = {}
    
    if redis_client:
        try:
            redis_client.ping()
            redis_status = "connected"
            storage_type = "redis"
        except:
            redis_status = "disconnected"
    
    if redis_status == "disconnected":
        if file_storage:
            storage_type = "file-based"
            storage_stats = file_storage.get_storage_stats()
        else:
            storage_type = "in-memory"
            # Count in-memory storage stats
            total_users = len(in_memory_storage)
            total_sessions = sum(len(user_data) for user_data in in_memory_storage.values())
            total_messages = sum(
                len(session_data["messages"]) 
                for user_data in in_memory_storage.values() 
                for session_data in user_data.values()
            )
            storage_stats = {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_messages": total_messages
            }
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "storage": storage_type,
        "persistent": storage_type in ["redis", "file-based"],
        "stats": storage_stats,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/storage/stats")
async def get_storage_statistics():
    """Get detailed storage statistics"""
    try:
        if redis_client:
            # Redis statistics would require custom implementation
            return {"storage_type": "redis", "message": "Redis stats not implemented yet"}
        elif file_storage:
            return file_storage.get_storage_stats()
        else:
            # In-memory statistics
            total_users = len(in_memory_storage)
            total_sessions = 0
            total_messages = 0
            user_details = []
            
            for email_hash, user_data in in_memory_storage.items():
                user_sessions = len(user_data)
                user_messages = sum(len(session_data["messages"]) for session_data in user_data.values())
                total_sessions += user_sessions
                total_messages += user_messages
                
                user_details.append({
                    "user_hash": email_hash[:8] + "...",
                    "sessions": user_sessions,
                    "messages": user_messages
                })
            
            return {
                "storage_type": "in-memory",
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "persistent": False,
                "users": user_details
            }
            
    except Exception as e:
        logger.error(f"Error getting storage stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get storage statistics")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8003))
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
