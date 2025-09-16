#!/usr/bin/env python3

"""PyAudio-based Porcupine wake word detection test.

This script uses PyAudio to capture microphone input, processes it through
the PorcupineFilter for wake word detection, and passes the audio through
to the speakers.

Requirements:
- pip install pvporcupine~=3.0.0
- export PICOVOICE_ACCESS_KEY="your_access_key"

Usage: python test_porcupine_microphone.py
"""

import asyncio
import os
import sys
import threading
import time
from queue import Queue

import numpy as np
import pyaudio

from pipecat.audio.filters.porcupine_filter import PorcupineFilter
from pipecat.frames.frames import WakeWordDetectionFrame


class PyAudioPorcupineTest:
    """Test class for Porcupine wake word detection with PyAudio."""

    def __init__(self, access_key: str):
        self.access_key = access_key
        self.porcupine_filter = None
        self.audio_queue = Queue()
        self.running = False
        
        # Audio settings
        self.chunk_size = 1024
        self.sample_rate = 16000  # Porcupine typically uses 16kHz
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # PyAudio instance
        self.pyaudio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

    async def initialize_porcupine(self):
        """Initialize the Porcupine filter."""
        try:
            self.porcupine_filter = PorcupineFilter(
                access_key=self.access_key,
                keywords=["picovoice", "bumblebee"],  # Free built-in keywords
                sensitivities=[0.5, 0.5]
            )
            
            # Use Porcupine's required sample rate
            self.sample_rate = self.porcupine_filter.sample_rate
            self.chunk_size = self.porcupine_filter.frame_length
            
            await self.porcupine_filter.start(self.sample_rate)
            
            print(f"✅ Porcupine initialized successfully")
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"   Frame length: {self.chunk_size} samples")
            print(f"   Keywords: {self.porcupine_filter.keywords}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Porcupine: {e}")
            return False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input callback - captures microphone data."""
        if status:
            print(f"Audio input status: {status}")
        
        # Put audio data in queue for processing
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def setup_audio_streams(self):
        """Set up PyAudio input and output streams."""
        try:
            # Input stream (microphone)
            self.input_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Output stream (speakers) - for audio passthrough
            self.output_stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("✅ Audio streams initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup audio streams: {e}")
            return False

    async def process_audio_loop(self):
        """Main audio processing loop."""
        print("🎙️  Starting audio processing...")
        print("💡 Say 'picovoice' or 'bumblebee' to test wake word detection")
        print("🛑 Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Get audio data from queue (non-blocking)
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Process through Porcupine filter
                    filtered_audio = await self.porcupine_filter.filter(audio_data)
                    
                    # Check for wake word detection
                    if hasattr(self.porcupine_filter, '_last_detection') and self.porcupine_filter._last_detection:
                        detection = self.porcupine_filter._last_detection
                        print(f"\n🎯 WAKE WORD DETECTED: '{detection.keyword}' (index: {detection.keyword_index})")
                        print(f"   Timestamp: {time.strftime('%H:%M:%S')}")
                        
                        # Clear the detection
                        self.porcupine_filter._last_detection = None
                    
                    # Pass audio through to speakers (optional - comment out if too loud)
                    # self.output_stream.write(filtered_audio)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
                break

    def start_streams(self):
        """Start audio streams."""
        if self.input_stream:
            self.input_stream.start_stream()
        if self.output_stream:
            self.output_stream.start_stream()

    def stop_streams(self):
        """Stop and close audio streams."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

    async def run(self):
        """Main run method."""
        # Initialize Porcupine
        if not await self.initialize_porcupine():
            return False
        
        # Setup audio streams
        if not self.setup_audio_streams():
            return False
        
        # Start audio streams
        self.start_streams()
        self.running = True
        
        try:
            # Run audio processing loop
            await self.process_audio_loop()
        except KeyboardInterrupt:
            print("\n👋 Stopping wake word detection...")
        finally:
            self.running = False
            self.stop_streams()
            await self.porcupine_filter.stop()
            self.pyaudio.terminate()
        
        return True

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'pyaudio') and self.pyaudio:
            self.pyaudio.terminate()


async def main():
    """Main function."""
    # Check for access key
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("❌ Please set PICOVOICE_ACCESS_KEY environment variable")
        print("   Get your access key from https://console.picovoice.ai/")
        sys.exit(1)
    
    # Create and run test
    test = PyAudioPorcupineTest(access_key)
    success = await test.run()
    
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed")
        sys.exit(1)


if __name__ == "__main__":
    print("🧪 PyAudio Porcupine Wake Word Detection Test")
    print("=" * 50)
    asyncio.run(main())
