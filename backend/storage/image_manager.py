import os
import uuid
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import logging
from datetime import datetime
import sqlite3
from contextlib import contextmanager

from config.settings import INPUTS_DIR, OUTPUTS_DIR, INTERMEDIATE_DIR, DATABASE_URL

logger = logging.getLogger(__name__)

class ImageManager:
    """
    Manages image storage, metadata, and database operations.
    
    This class handles:
    - Image file management
    - Metadata storage and retrieval
    - Database operations
    - Session management
    """
    
    def __init__(self):
        self.db_path = DATABASE_URL.replace("sqlite:///", "")
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create images table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        original_filename TEXT,
                        file_path TEXT NOT NULL,
                        width INTEGER,
                        height INTEGER,
                        mode TEXT,
                        file_size INTEGER,
                        image_type TEXT,
                        session_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        original_image_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        num_edits INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)
                
                # Create edits table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS edits (
                        edit_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        step_id INTEGER NOT NULL,
                        input_image_id TEXT NOT NULL,
                        output_image_id TEXT NOT NULL,
                        instruction TEXT NOT NULL,
                        metrics TEXT,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                        FOREIGN KEY (input_image_id) REFERENCES images (id),
                        FOREIGN KEY (output_image_id) REFERENCES images (id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def store_image(
        self,
        image_file_path: str,
        original_filename: str = None,
        image_type: str = "input",
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store an image and its metadata.
        
        Args:
            image_file_path: Path to the image file
            original_filename: Original filename
            image_type: Type of image (input, output, intermediate)
            session_id: Session ID if associated with a session
            metadata: Additional metadata
            
        Returns:
            Image ID
        """
        try:
            # Generate unique ID
            image_id = str(uuid.uuid4())
            
            # Determine target directory
            if image_type == "input":
                target_dir = INPUTS_DIR
            elif image_type == "output":
                target_dir = OUTPUTS_DIR
            elif image_type == "intermediate":
                target_dir = INTERMEDIATE_DIR
            else:
                raise ValueError(f"Invalid image type: {image_type}")
            
            # Get file extension
            source_path = Path(image_file_path)
            file_extension = source_path.suffix.lower()
            filename = f"{image_id}{file_extension}"
            target_path = target_dir / filename
            
            # Copy file to target location
            shutil.copy2(image_file_path, target_path)
            
            # Get image metadata
            with Image.open(target_path) as img:
                width, height = img.size
                mode = img.mode
            
            file_size = target_path.stat().st_size
            
            # Store in database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO images 
                    (id, filename, original_filename, file_path, width, height, mode, 
                     file_size, image_type, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_id, filename, original_filename, str(target_path),
                    width, height, mode, file_size, image_type, session_id,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
            
            logger.info(f"Image stored: {image_id} ({image_type})")
            return image_id
            
        except Exception as e:
            logger.error(f"Failed to store image: {str(e)}")
            raise
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get image information by ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Image information dictionary or None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
                row = cursor.fetchone()
                
                if row:
                    info = dict(row)
                    if info['metadata']:
                        info['metadata'] = json.loads(info['metadata'])
                    return info
                return None
                
        except Exception as e:
            logger.error(f"Failed to get image info: {str(e)}")
            return None
    
    def get_image_path(self, image_id: str) -> Optional[Path]:
        """
        Get the file path for an image ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Path to image file or None
        """
        try:
            info = self.get_image_info(image_id)
            if info:
                return Path(info['file_path'])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get image path: {str(e)}")
            return None
    
    def create_session(self, original_image_id: str = None) -> str:
        """
        Create a new editing session.
        
        Args:
            original_image_id: ID of the original image
            
        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (session_id, original_image_id)
                    VALUES (?, ?)
                """, (session_id, original_image_id))
                conn.commit()
            
            logger.info(f"Session created: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information or None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                
                if row:
                    info = dict(row)
                    if info['metadata']:
                        info['metadata'] = json.loads(info['metadata'])
                    return info
                return None
                
        except Exception as e:
            logger.error(f"Failed to get session info: {str(e)}")
            return None
    
    def add_edit(
        self,
        session_id: str,
        step_id: int,
        input_image_id: str,
        output_image_id: str,
        instruction: str,
        metrics: Dict[str, float] = None,
        processing_time: float = 0.0
    ) -> str:
        """
        Add an edit to a session.
        
        Args:
            session_id: Session ID
            step_id: Step number
            input_image_id: Input image ID
            output_image_id: Output image ID
            instruction: Text instruction
            metrics: Evaluation metrics
            processing_time: Processing time in seconds
            
        Returns:
            Edit ID
        """
        try:
            edit_id = str(uuid.uuid4())
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Add edit
                cursor.execute("""
                    INSERT INTO edits 
                    (edit_id, session_id, step_id, input_image_id, output_image_id,
                     instruction, metrics, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    edit_id, session_id, step_id, input_image_id, output_image_id,
                    instruction, json.dumps(metrics) if metrics else None, processing_time
                ))
                
                # Update session
                cursor.execute("""
                    UPDATE sessions 
                    SET last_updated = CURRENT_TIMESTAMP, num_edits = num_edits + 1
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
            
            logger.info(f"Edit added: {edit_id} to session {session_id}")
            return edit_id
            
        except Exception as e:
            logger.error(f"Failed to add edit: {str(e)}")
            raise
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get edit history for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of edit steps
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM edits 
                    WHERE session_id = ? 
                    ORDER BY step_id ASC
                """, (session_id,))
                
                edits = []
                for row in cursor.fetchall():
                    edit = dict(row)
                    if edit['metrics']:
                        edit['metrics'] = json.loads(edit['metrics'])
                    edits.append(edit)
                
                return edits
                
        except Exception as e:
            logger.error(f"Failed to get session history: {str(e)}")
            return []
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session information
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM sessions 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    session = dict(row)
                    if session['metadata']:
                        session['metadata'] = json.loads(session['metadata'])
                    sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to list sessions: {str(e)}")
            return []
    
    def cleanup_old_files(self, days_old: int = 7) -> int:
        """
        Clean up old image files and database entries.
        
        Args:
            days_old: Age in days to consider files old
            
        Returns:
            Number of files cleaned up
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
            cleaned_count = 0
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Find old images
                cursor.execute("""
                    SELECT id, file_path FROM images 
                    WHERE created_at < datetime(?, 'unixepoch')
                """, (cutoff_date,))
                
                for row in cursor.fetchall():
                    image_id, file_path = row
                    
                    # Delete file
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except FileNotFoundError:
                        pass
                    
                    # Delete database entry
                    cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
                
                conn.commit()
            
            logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {str(e)}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics
        """
        try:
            stats = {}
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Count images by type
                cursor.execute("""
                    SELECT image_type, COUNT(*) as count 
                    FROM images 
                    GROUP BY image_type
                """)
                stats['images_by_type'] = dict(cursor.fetchall())
                
                # Count sessions
                cursor.execute("SELECT COUNT(*) as count FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                # Count edits
                cursor.execute("SELECT COUNT(*) as count FROM edits")
                stats['total_edits'] = cursor.fetchone()[0]
            
            # Calculate file sizes
            stats['storage_size'] = {
                'inputs': self._get_dir_size(INPUTS_DIR),
                'outputs': self._get_dir_size(OUTPUTS_DIR),
                'intermediate': self._get_dir_size(INTERMEDIATE_DIR)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    def _get_dir_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        try:
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
