#!/usr/bin/env python3
"""
Test script for STT tracking functionality
"""

import asyncio
import aiohttp
import json
import base64
import wave
import io
import numpy as np
from datetime import datetime

# Generate a simple test audio file (WAV format)
def generate_test_audio():
    """Generate a simple test WAV audio file"""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

async def test_stt_tracking():
    """Test STT tracking functionality"""
    base_url = "http://localhost:4000"  # Adjust port if needed
    
    # Generate test audio
    test_audio = generate_test_audio()
    print(f"Generated test audio: {len(test_audio)} bytes")
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: File upload transcription
        print("\n=== Test 1: File Upload Transcription ===")
        try:
            data = aiohttp.FormData()
            data.add_field('audio_file', test_audio, filename='test.wav', content_type='audio/wav')
            data.add_field('language_code', 'en-US')
            data.add_field('user_id', 'test_user_123')
            
            async with session.post(f"{base_url}/api/stt/transcribe", data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Transcription successful: {result.get('transcript', 'No transcript')}")
                else:
                    print(f"❌ Transcription failed: {resp.status}")
                    print(await resp.text())
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
        
        # Test 2: Base64 transcription
        print("\n=== Test 2: Base64 Transcription ===")
        try:
            audio_base64 = base64.b64encode(test_audio).decode('utf-8')
            
            data = aiohttp.FormData()
            data.add_field('audio_base64', audio_base64)
            data.add_field('language_code', 'en-US')
            data.add_field('encoding', 'LINEAR16')
            data.add_field('sample_rate_hertz', '16000')
            data.add_field('user_id', 'test_user_456')
            
            async with session.post(f"{base_url}/api/stt/transcribe-base64", data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Base64 transcription successful: {result.get('transcript', 'No transcript')}")
                else:
                    print(f"❌ Base64 transcription failed: {resp.status}")
                    print(await resp.text())
        except Exception as e:
            print(f"❌ Test 2 failed: {e}")
        
        # Wait a moment for data to be saved
        await asyncio.sleep(1)
        
        # Test 3: Check usage summary
        print("\n=== Test 3: Usage Summary ===")
        try:
            async with session.get(f"{base_url}/api/stt/usage/summary?days=1") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get('success'):
                        summary = result['data']['summary']
                        print(f"✅ Usage summary retrieved:")
                        print(f"   Total requests: {summary.get('total_requests', 0)}")
                        print(f"   Total audio duration: {summary.get('total_audio_duration_hours', 0)} hours")
                        print(f"   Total audio size: {summary.get('total_audio_size_mb', 0)} MB")
                        print(f"   Success rate: {summary.get('success_rate_percent', 0)}%")
                    else:
                        print(f"❌ Usage summary failed: {result}")
                else:
                    print(f"❌ Usage summary request failed: {resp.status}")
                    print(await resp.text())
        except Exception as e:
            print(f"❌ Test 3 failed: {e}")
        
        # Test 4: Check detailed usage
        print("\n=== Test 4: Detailed Usage ===")
        try:
            async with session.get(f"{base_url}/api/stt/usage/detailed?days=1&limit=10") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if result.get('success'):
                        records = result['data']
                        print(f"✅ Detailed usage retrieved: {len(records)} records")
                        for i, record in enumerate(records[:3]):  # Show first 3 records
                            print(f"   Record {i+1}:")
                            print(f"     Endpoint: {record.get('endpoint', 'N/A')}")
                            print(f"     Duration: {record.get('audio_duration_seconds', 0):.2f}s")
                            print(f"     Size: {record.get('audio_file_size_bytes', 0)} bytes")
                            print(f"     Success: {record.get('success', False)}")
                    else:
                        print(f"❌ Detailed usage failed: {result}")
                else:
                    print(f"❌ Detailed usage request failed: {resp.status}")
        except Exception as e:
            print(f"❌ Test 4 failed: {e}")
        
        # Test 5: Test dashboard access
        print("\n=== Test 5: Dashboard Access ===")
        try:
            async with session.get(f"{base_url}/api/stt/dashboard") as resp:
                if resp.status == 200:
                    print(f"✅ Dashboard accessible at: {base_url}/api/stt/dashboard")
                else:
                    print(f"❌ Dashboard not accessible: {resp.status}")
        except Exception as e:
            print(f"❌ Test 5 failed: {e}")

def main():
    """Main test function"""
    print("STT Tracking Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    try:
        asyncio.run(test_stt_tracking())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest suite failed: {e}")
    
    print(f"\nTest completed at: {datetime.now()}")
    print("\nTo view the dashboard, visit: http://localhost:4000/api/stt/dashboard")

if __name__ == "__main__":
    main()