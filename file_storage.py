"""
File-based storage system for chat history persistence
This provides permanent storage when Redis is not available
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileStorage:
    def __init__(self, storage_dir: str = "chat_storage"):
        self.storage_dir = storage_dir
        self.ensure_storage_directory()
    
    def ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created storage directory: {self.storage_dir}")
    
    def get_user_file_path(self, gmail: str) -> str:
        """Generate file path for user's chat data"""
        email_hash = hashlib.md5(gmail.lower().encode()).hexdigest()
        return os.path.join(self.storage_dir, f"user_{email_hash}.json")
    
    def load_user_data(self, gmail: str) -> Dict[str, Any]:
        """Load user's chat data from file"""
        file_path = self.get_user_file_path(gmail)
        
        if not os.path.exists(file_path):
            return {"sessions": {}, "created_at": datetime.now(timezone.utc).isoformat()}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading user data for {gmail}: {str(e)}")
            return {"sessions": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    
    def save_user_data(self, gmail: str, data: Dict[str, Any]):
        """Save user's chat data to file"""
        file_path = self.get_user_file_path(gmail)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving user data for {gmail}: {str(e)}")
    
    def save_message(self, gmail: str, session_id: str, role: str, content: str):
        """Save a chat message to file storage"""
        try:
            data = self.load_user_data(gmail)
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Initialize session if it doesn't exist
            if session_id not in data["sessions"]:
                data["sessions"][session_id] = {
                    "messages": [],
                    "created_at": timestamp,
                    "last_updated": timestamp
                }
            
            # Add message
            message = {
                "role": role,
                "content": content,
                "timestamp": timestamp
            }
            
            data["sessions"][session_id]["messages"].append(message)
            data["sessions"][session_id]["last_updated"] = timestamp
            
            self.save_user_data(gmail, data)
            logger.info(f"Saved message for {gmail} in session {session_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
    
    def get_chat_history(self, gmail: str, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve chat history for a specific session"""
        try:
            data = self.load_user_data(gmail)
            
            if session_id not in data["sessions"]:
                return []
            
            messages = data["sessions"][session_id]["messages"]
            return messages[-limit:] if limit else messages
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def get_user_sessions(self, gmail: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        try:
            data = self.load_user_data(gmail)
            sessions = []
            
            for session_id, session_data in data["sessions"].items():
                sessions.append({
                    "session_id": session_id,
                    "last_updated": session_data.get("last_updated", session_data.get("created_at", "")),
                    "message_count": len(session_data.get("messages", []))
                })
            
            # Sort by last updated (most recent first)
            sessions.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving user sessions: {str(e)}")
            return []
    
    def delete_session(self, gmail: str, session_id: str) -> bool:
        """Delete a specific chat session"""
        try:
            data = self.load_user_data(gmail)
            
            if session_id in data["sessions"]:
                del data["sessions"][session_id]
                self.save_user_data(gmail, data)
                logger.info(f"Deleted session {session_id[:8]}... for {gmail}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            files = [f for f in os.listdir(self.storage_dir) if f.startswith("user_") and f.endswith(".json")]
            total_users = len(files)
            total_sessions = 0
            total_messages = 0
            
            for file in files:
                file_path = os.path.join(self.storage_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions = data.get("sessions", {})
                        total_sessions += len(sessions)
                        for session_data in sessions.values():
                            total_messages += len(session_data.get("messages", []))
                except:
                    continue
            
            return {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "storage_type": "file_based"
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e), "storage_type": "file_based"}
