from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from config.settings import INTERMEDIATE_DIR, OUTPUTS_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

class EditStep(BaseModel):
    step_id: int
    image_id: str
    instruction: str
    output_image_id: str
    metrics: Dict[str, float]
    timestamp: str
    processing_time: float

class EditHistory(BaseModel):
    session_id: str
    original_image_id: str
    steps: List[EditStep]
    created_at: str
    last_updated: str

# In-memory storage for sessions (in production, use database)
sessions: Dict[str, EditHistory] = {}

@router.post("/session")
async def create_session():
    """
    Create a new editing session.
    
    Returns:
        New session information
    """
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        if session_id not in sessions:
            sessions[session_id] = EditHistory(
                session_id=session_id,
                original_image_id="",
                steps=[],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        
        logger.info(f"Created new session: {session_id}")
        return {"session_id": session_id, "message": "Session created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/history/{session_id}", response_model=EditHistory)
async def get_edit_history(session_id: str):
    """
    Get edit history for a session.
    
    Args:
        session_id: Session ID to retrieve history for
        
    Returns:
        Edit history for the session
    """
    try:
        if session_id not in sessions:
            # Create new session if it doesn't exist
            sessions[session_id] = EditHistory(
                session_id=session_id,
                original_image_id="",
                steps=[],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        
        return sessions[session_id]
        
    except Exception as e:
        logger.error(f"Error getting history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/history/{session_id}/step")
async def add_edit_step(session_id: str, step: EditStep):
    """
    Add an edit step to session history.
    
    Args:
        session_id: Session ID
        step: Edit step to add
        
    Returns:
        Updated session history
    """
    try:
        if session_id not in sessions:
            sessions[session_id] = EditHistory(
                session_id=session_id,
                original_image_id=step.image_id if step.step_id == 0 else "",
                steps=[],
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        
        # Add the step
        sessions[session_id].steps.append(step)
        sessions[session_id].last_updated = datetime.now().isoformat()
        
        logger.info(f"Added step {step.step_id} to session {session_id}")
        
        return sessions[session_id]
        
    except Exception as e:
        logger.error(f"Error adding step to session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add step: {str(e)}")

@router.get("/history/{session_id}/steps/{step_id}")
async def get_edit_step(session_id: str, step_id: int):
    """
    Get a specific edit step from session history.
    
    Args:
        session_id: Session ID
        step_id: Step ID to retrieve
        
    Returns:
        Specific edit step
    """
    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        session = sessions[session_id]
        
        # Find the step
        step = None
        for s in session.steps:
            if s.step_id == step_id:
                step = s
                break
        
        if not step:
            raise HTTPException(
                status_code=404,
                detail=f"Step {step_id} not found in session {session_id}"
            )
        
        return step
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting step {step_id} from session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get step: {str(e)}")

@router.delete("/history/{session_id}")
async def delete_session_history(session_id: str):
    """
    Delete session history.
    
    Args:
        session_id: Session ID to delete
        
    Returns:
        Deletion status
    """
    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        del sessions[session_id]
        logger.info(f"Deleted session history: {session_id}")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.get("/history")
async def list_sessions():
    """
    List all active sessions.
    
    Returns:
        List of session summaries
    """
    try:
        session_summaries = []
        for session_id, session in sessions.items():
            session_summaries.append({
                "session_id": session_id,
                "original_image_id": session.original_image_id,
                "num_steps": len(session.steps),
                "created_at": session.created_at,
                "last_updated": session.last_updated
            })
        
        return {"sessions": session_summaries}
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
