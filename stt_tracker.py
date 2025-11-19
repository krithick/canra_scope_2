from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from database import get_db
import uuid
import wave
import io

class STTUsageRecord(BaseModel):
    """Model for tracking STT usage"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    endpoint: str  # Which STT endpoint was used
    audio_duration_seconds: float = 0.0
    audio_file_size_bytes: int = 0
    language_code: str = "en-US"
    encoding: str = "WEBM_OPUS"
    sample_rate: int = 16000
    transcript_length: int = 0  # Number of characters in transcript
    word_count: int = 0
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None

class STTTracker:
    """Service for tracking STT usage and generating reports"""
    
    def __init__(self, db):
        self.db = db
    
    def _calculate_audio_duration(self, audio_content: bytes, encoding: str, sample_rate: int) -> float:
        """Calculate audio duration from audio content"""
        try:
            if encoding == "LINEAR16" and audio_content.startswith(b'RIFF'):
                # WAV file - can calculate exact duration
                with wave.open(io.BytesIO(audio_content), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    framerate = wav_file.getframerate()
                    return frames / framerate
            else:
                # For other formats, estimate based on file size and bitrate
                # This is an approximation - actual duration may vary
                if encoding == "WEBM_OPUS":
                    # Opus typically uses ~32kbps for speech
                    estimated_bitrate = 32000  # bits per second
                elif encoding == "FLAC":
                    # FLAC varies but roughly 500-1000 kbps for speech
                    estimated_bitrate = 750000
                else:
                    # Default estimation
                    estimated_bitrate = 128000
                
                # Convert bytes to bits and divide by bitrate
                duration = (len(audio_content) * 8) / estimated_bitrate
                return max(duration, 0.1)  # Minimum 0.1 seconds
                
        except Exception:
            # Fallback estimation based on file size
            # Very rough estimate: assume 16kHz, 16-bit mono = 32KB/second
            return len(audio_content) / 32000
    
    async def track_transcription(
        self,
        endpoint: str,
        audio_content: bytes,
        config: dict,
        result: dict,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """Track a transcription request"""
        
        # Calculate audio duration
        duration = self._calculate_audio_duration(
            audio_content, 
            config.get('encoding', 'WEBM_OPUS'),
            config.get('sample_rate_hertz', 16000)
        )
        
        # Create usage record
        usage_record = STTUsageRecord(
            session_id=session_id,
            endpoint=endpoint,
            audio_duration_seconds=duration,
            audio_file_size_bytes=len(audio_content),
            language_code=config.get('language_code', 'en-US'),
            encoding=config.get('encoding', 'WEBM_OPUS'),
            sample_rate=config.get('sample_rate_hertz', 16000),
            transcript_length=len(result.get('transcript', '')) if success else 0,
            word_count=result.get('word_count', 0) if success else 0,
            confidence_score=result.get('confidence', 0.0) if success else 0.0,
            processing_time_ms=result.get('processing_time_ms', 0) if success else 0,
            success=success,
            error_message=error_message,
            user_id=user_id
        )
        
        # Save to database
        await self.db.stt_usage.insert_one(usage_record.dict())
        
        return usage_record.id
    
    async def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> Dict:
        """Get usage summary for reporting"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Build query
        query = {
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        if user_id:
            query["user_id"] = user_id
        
        # Get all records in date range
        records = await self.db.stt_usage.find(query).to_list(length=None)
        
        if not records:
            return {
                "period": {"start": start_date, "end": end_date},
                "total_requests": 0,
                "total_audio_duration_hours": 0,
                "total_audio_size_mb": 0,
                "success_rate": 0,
                "endpoints": {},
                "languages": {},
                "daily_usage": []
            }
        
        # Calculate summary statistics
        total_requests = len(records)
        successful_requests = len([r for r in records if r['success']])
        total_duration = sum(r['audio_duration_seconds'] for r in records)
        total_size = sum(r['audio_file_size_bytes'] for r in records)
        
        # Endpoint breakdown
        endpoint_stats = {}
        for record in records:
            endpoint = record['endpoint']
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    "requests": 0,
                    "duration_seconds": 0,
                    "size_bytes": 0,
                    "success_count": 0
                }
            endpoint_stats[endpoint]["requests"] += 1
            endpoint_stats[endpoint]["duration_seconds"] += record['audio_duration_seconds']
            endpoint_stats[endpoint]["size_bytes"] += record['audio_file_size_bytes']
            if record['success']:
                endpoint_stats[endpoint]["success_count"] += 1
        
        # Language breakdown
        language_stats = {}
        for record in records:
            lang = record['language_code']
            if lang not in language_stats:
                language_stats[lang] = {"requests": 0, "duration_seconds": 0}
            language_stats[lang]["requests"] += 1
            language_stats[lang]["duration_seconds"] += record['audio_duration_seconds']
        
        # Daily usage breakdown
        daily_usage = {}
        for record in records:
            date_key = record['timestamp'].strftime('%Y-%m-%d')
            if date_key not in daily_usage:
                daily_usage[date_key] = {
                    "date": date_key,
                    "requests": 0,
                    "duration_seconds": 0,
                    "size_bytes": 0
                }
            daily_usage[date_key]["requests"] += 1
            daily_usage[date_key]["duration_seconds"] += record['audio_duration_seconds']
            daily_usage[date_key]["size_bytes"] += record['audio_file_size_bytes']
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate_percent": round((successful_requests / total_requests) * 100, 2),
                "total_audio_duration_hours": round(total_duration / 3600, 2),
                "total_audio_size_mb": round(total_size / (1024 * 1024), 2),
                "average_audio_duration_seconds": round(total_duration / total_requests, 2),
                "average_processing_time_ms": round(sum(r.get('processing_time_ms', 0) for r in records) / total_requests, 2)
            },
            "endpoints": endpoint_stats,
            "languages": language_stats,
            "daily_usage": list(daily_usage.values())
        }
    
    async def get_detailed_usage(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get detailed usage records"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        query = {
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        
        records = await self.db.stt_usage.find(query).sort("timestamp", -1).limit(limit).to_list(length=None)
        
        # Convert ObjectId to string for JSON serialization
        for record in records:
            if '_id' in record:
                record['_id'] = str(record['_id'])
        
        return records