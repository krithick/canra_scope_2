#!/usr/bin/env python3
"""
Test script for Google STT API integration
"""

import asyncio
import os
from dotenv import load_dotenv
from stt_service import GoogleSTTService
from models import STTRequest

# Load environment variables
load_dotenv('.env')

async def test_stt_service():
    """Test the STT service with a sample audio file"""
    
    try:
        # Initialize STT service
        print("Initializing Google STT service...")
        stt_service = GoogleSTTService(credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH"))
        print("✓ STT service initialized successfully")
        
        # Test supported languages
        print("\nTesting supported languages...")
        languages = stt_service.get_supported_languages()
        print(f"✓ Found {len(languages)} supported languages")
        print("Sample languages:", list(languages.items())[:5])
        
        # Test supported encodings
        print("\nTesting supported encodings...")
        encodings = stt_service.get_supported_encodings()
        print(f"✓ Found {len(encodings)} supported encodings")
        print("Sample encodings:", list(encodings.items())[:3])
        
        # Test with sample audio (you would need to provide an actual audio file)
        print("\nSTT service is ready for audio transcription!")
        print("To test with actual audio, upload a file through the API endpoints:")
        print("- POST /api/stt/transcribe")
        print("- POST /api/stt/transcribe-base64")
        print("- POST /api/stt/question-answer")
        print("- POST /api/stt/explanation")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing STT service: {e}")
        return False

async def test_configuration():
    """Test configuration and credentials"""
    
    print("Testing STT configuration...")
    
    # Check environment variables
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    
    print(f"Credentials path: {credentials_path}")
    print(f"Project ID: {project_id}")
    
    if not credentials_path:
        print("✗ GOOGLE_CREDENTIALS_PATH not set in .env file")
        return False
    
    if not project_id:
        print("✗ GOOGLE_PROJECT_ID not set in .env file")
        return False
    
    # Check if credentials file exists
    if not os.path.exists(credentials_path):
        print(f"✗ Credentials file not found: {credentials_path}")
        print("Please ensure the Google Cloud service account JSON file is in the project directory")
        return False
    
    print("✓ Configuration looks good")
    return True

def print_api_documentation():
    """Print API documentation for STT endpoints"""
    
    print("\n" + "="*60)
    print("GOOGLE STT API ENDPOINTS DOCUMENTATION")
    print("="*60)
    
    endpoints = [
        {
            "method": "POST",
            "path": "/api/stt/transcribe",
            "description": "Transcribe uploaded audio file to text",
            "parameters": [
                "audio_file: UploadFile (required) - Audio file to transcribe",
                "language_code: str (default: en-US) - Language code",
                "sample_rate_hertz: int (default: 16000) - Sample rate in Hz",
                "encoding: str (default: WEBM_OPUS) - Audio encoding format",
                "enable_automatic_punctuation: bool (default: True)",
                "model: str (default: latest_long) - Recognition model"
            ]
        },
        {
            "method": "POST", 
            "path": "/api/stt/transcribe-base64",
            "description": "Transcribe base64 encoded audio to text",
            "parameters": [
                "audio_base64: str (required) - Base64 encoded audio data",
                "language_code: str (default: en-US)",
                "sample_rate_hertz: int (default: 16000)",
                "encoding: str (default: WEBM_OPUS)"
            ]
        },
        {
            "method": "POST",
            "path": "/api/stt/question-answer", 
            "description": "Transcribe audio answer and submit to question session",
            "parameters": [
                "session_id: str (required) - Question session ID",
                "audio_file: UploadFile (required) - Audio with user's answer",
                "language_code: str (default: en-US)",
                "time_taken: int (default: 30) - Time taken to answer"
            ]
        },
        {
            "method": "POST",
            "path": "/api/stt/explanation",
            "description": "Transcribe audio explanation and submit to question session", 
            "parameters": [
                "session_id: str (required) - Question session ID",
                "audio_file: UploadFile (required) - Audio with explanation",
                "language_code: str (default: en-US)"
            ]
        },
        {
            "method": "GET",
            "path": "/api/stt/supported-languages",
            "description": "Get list of supported languages for STT",
            "parameters": []
        },
        {
            "method": "GET", 
            "path": "/api/stt/supported-encodings",
            "description": "Get list of supported audio encodings for STT",
            "parameters": []
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n{endpoint['method']} {endpoint['path']}")
        print(f"Description: {endpoint['description']}")
        if endpoint['parameters']:
            print("Parameters:")
            for param in endpoint['parameters']:
                print(f"  - {param}")
        else:
            print("Parameters: None")
    
    print("\n" + "="*60)
    print("RESPONSE FORMAT")
    print("="*60)
    print("""
STTResponse Model:
{
    "transcript": "string",
    "confidence": 0.95,
    "language_code": "en-US", 
    "processing_time_ms": 1500,
    "word_count": 10,
    "alternatives": [
        {
            "transcript": "alternative text",
            "confidence": 0.85
        }
    ]
}
    """)

async def main():
    """Main test function"""
    
    print("Google STT API Integration Test")
    print("="*40)
    
    # Test configuration
    config_ok = await test_configuration()
    if not config_ok:
        print("\n✗ Configuration test failed. Please fix the issues above.")
        return
    
    # Test STT service
    stt_ok = await test_stt_service()
    if not stt_ok:
        print("\n✗ STT service test failed.")
        return
    
    print("\n✓ All tests passed! STT API is ready to use.")
    
    # Print API documentation
    print_api_documentation()

if __name__ == "__main__":
    asyncio.run(main())