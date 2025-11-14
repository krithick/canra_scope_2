import os
import time
from typing import Optional, Dict, Any
from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
import io
from models import STTRequest, STTResponse, STTStreamRequest
from fastapi import HTTPException
import logging
import wave

logger = logging.getLogger(__name__)

class GoogleSTTService:
    """Google Speech-to-Text service wrapper"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Google STT client
        
        :param credentials_path: Path to Google Cloud credentials JSON file
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = speech.SpeechClient()
            logger.info("Google STT client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google STT client: {e}")
            raise
    
    def _get_encoding(self, encoding_str: str) -> speech.RecognitionConfig.AudioEncoding:
        """Convert string encoding to Google STT encoding enum"""
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "FLAC": speech.RecognitionConfig.AudioEncoding.FLAC,
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
            "AMR": speech.RecognitionConfig.AudioEncoding.AMR,
            "AMR_WB": speech.RecognitionConfig.AudioEncoding.AMR_WB,
            "OGG_OPUS": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "SPEEX_WITH_HEADER_BYTE": speech.RecognitionConfig.AudioEncoding.SPEEX_WITH_HEADER_BYTE,
            "WEBM_OPUS": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        }
        return encoding_map.get(encoding_str, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
    
    def _detect_audio_format(self, audio_content: bytes) -> tuple[str, int]:
        """Detect audio format and sample rate from audio content"""
        try:
            # Check if it's a WAV file
            if audio_content.startswith(b'RIFF') and b'WAVE' in audio_content[:12]:
                # Parse WAV header
                with wave.open(io.BytesIO(audio_content), 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    
                    # For WAV files, use LINEAR16 encoding
                    return "LINEAR16", sample_rate
            
            # Default for other formats
            return "WEBM_OPUS", 16000
            
        except Exception:
            # Fallback to default
            return "WEBM_OPUS", 16000
    
    async def transcribe_audio(self, audio_content: bytes, config: STTRequest) -> STTResponse:
        """
        Transcribe audio content to text
        
        :param audio_content: Raw audio bytes
        :param config: STT configuration
        :return: Transcription response
        """
        start_time = time.time()
        
        try:
            # Auto-detect format for better compatibility
            detected_encoding, detected_sample_rate = self._detect_audio_format(audio_content)
            
            # Use detected values if they seem more appropriate
            final_encoding = detected_encoding if detected_encoding == "LINEAR16" else config.encoding
            final_sample_rate = detected_sample_rate if detected_encoding == "LINEAR16" else config.sample_rate_hertz
            
            # Configure recognition
            recognition_config = RecognitionConfig(
                encoding=self._get_encoding(final_encoding),
                sample_rate_hertz=final_sample_rate,
                language_code=config.language_code,
                enable_automatic_punctuation=config.enable_automatic_punctuation,
                enable_word_time_offsets=config.enable_word_time_offsets,
                model=config.model,
                max_alternatives=3  # Get up to 3 alternative transcriptions
            )
            
            # Create audio object
            audio = RecognitionAudio(content=audio_content)
            
            # Perform recognition
            response = self.client.recognize(
                config=recognition_config,
                audio=audio
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if not response.results:
                return STTResponse(
                    transcript="",
                    confidence=0.0,
                    language_code=config.language_code,
                    processing_time_ms=processing_time,
                    word_count=0,
                    alternatives=[]
                )
            
            # Get the best result
            result = response.results[0]
            alternative = result.alternatives[0]
            
            # Prepare alternatives
            alternatives = []
            for alt in result.alternatives:
                alternatives.append({
                    "transcript": alt.transcript,
                    "confidence": alt.confidence
                })
            
            return STTResponse(
                transcript=alternative.transcript,
                confidence=alternative.confidence,
                language_code=config.language_code,
                processing_time_ms=processing_time,
                word_count=len(alternative.transcript.split()),
                alternatives=alternatives
            )
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    async def transcribe_long_audio(self, audio_content: bytes, config: STTRequest) -> STTResponse:
        """
        Transcribe long audio files using long-running operation
        
        :param audio_content: Raw audio bytes
        :param config: STT configuration
        :return: Transcription response
        """
        start_time = time.time()
        
        try:
            # Auto-detect format for better compatibility
            detected_encoding, detected_sample_rate = self._detect_audio_format(audio_content)
            
            # Use detected values if they seem more appropriate
            final_encoding = detected_encoding if detected_encoding == "LINEAR16" else config.encoding
            final_sample_rate = detected_sample_rate if detected_encoding == "LINEAR16" else config.sample_rate_hertz
            
            # For long audio, we need to use long_running_recognize
            recognition_config = RecognitionConfig(
                encoding=self._get_encoding(final_encoding),
                sample_rate_hertz=final_sample_rate,
                language_code=config.language_code,
                enable_automatic_punctuation=config.enable_automatic_punctuation,
                enable_word_time_offsets=config.enable_word_time_offsets,
                model=config.model
            )
            
            audio = RecognitionAudio(content=audio_content)
            
            # Start long-running operation
            operation = self.client.long_running_recognize(
                config=recognition_config,
                audio=audio
            )
            
            # Wait for operation to complete
            response = operation.result(timeout=300)  # 5 minute timeout
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if not response.results:
                return STTResponse(
                    transcript="",
                    confidence=0.0,
                    language_code=config.language_code,
                    processing_time_ms=processing_time,
                    word_count=0,
                    alternatives=[]
                )
            
            # Combine all results
            full_transcript = ""
            total_confidence = 0.0
            alternatives = []
            
            for result in response.results:
                alternative = result.alternatives[0]
                full_transcript += alternative.transcript + " "
                total_confidence += alternative.confidence
                
                alternatives.append({
                    "transcript": alternative.transcript,
                    "confidence": alternative.confidence
                })
            
            avg_confidence = total_confidence / len(response.results) if response.results else 0.0
            
            return STTResponse(
                transcript=full_transcript.strip(),
                confidence=avg_confidence,
                language_code=config.language_code,
                processing_time_ms=processing_time,
                word_count=len(full_transcript.split()),
                alternatives=alternatives
            )
            
        except Exception as e:
            logger.error(f"Long audio STT transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Long audio transcription failed: {str(e)}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return {
            "en-US": "English (United States)",
            "en-GB": "English (United Kingdom)",
            "es-ES": "Spanish (Spain)",
            "es-US": "Spanish (United States)",
            "fr-FR": "French (France)",
            "de-DE": "German (Germany)",
            "it-IT": "Italian (Italy)",
            "pt-BR": "Portuguese (Brazil)",
            "ru-RU": "Russian (Russia)",
            "ja-JP": "Japanese (Japan)",
            "ko-KR": "Korean (South Korea)",
            "zh-CN": "Chinese (Simplified)",
            "hi-IN": "Hindi (India)",
            "ar-SA": "Arabic (Saudi Arabia)"
        }
    
    def get_supported_encodings(self) -> Dict[str, str]:
        """Get list of supported audio encodings"""
        return {
            "LINEAR16": "Linear PCM 16-bit",
            "FLAC": "FLAC (Free Lossless Audio Codec)",
            "MULAW": "μ-law",
            "AMR": "Adaptive Multi-Rate",
            "AMR_WB": "Adaptive Multi-Rate Wideband",
            "OGG_OPUS": "Ogg Opus",
            "SPEEX_WITH_HEADER_BYTE": "Speex with header byte",
            "WEBM_OPUS": "WebM Opus"
        }