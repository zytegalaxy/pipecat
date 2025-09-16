#!/usr/bin/env python3

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Example demonstrating Porcupine wake word detection with Pipecat.

This example shows how to use the PorcupineFilter to detect wake words
in audio streams. When a wake word is detected, it triggers a response.

To run this example:
1. Get a Picovoice access key from https://console.picovoice.ai/
2. Install porcupine: pip install pipecat-ai[porcupine]
3. Set your access key: export PICOVOICE_ACCESS_KEY=your_key_here
4. Run: python 17-porcupine-wake-word-detection.py
"""

import asyncio
import os
import sys

from loguru import logger

from pipecat.audio.filters.porcupine_filter import PorcupineFilter
from pipecat.frames.frames import WakeWordDetectionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure


class WakeWordHandler(FrameProcessor):
    """Frame processor that handles wake word detection events."""

    def __init__(self):
        super().__init__()
        self._wake_word_detected = False

    async def process_frame(self, frame, direction):
        """Process frames and handle wake word detection."""
        if isinstance(frame, WakeWordDetectionFrame):
            logger.info(f"🎯 Wake word detected: '{frame.keyword}' (confidence: {frame.confidence})")
            self._wake_word_detected = True
            
            # You can trigger actions here, such as:
            # - Start listening for voice commands
            # - Play a confirmation sound
            # - Activate the voice assistant
            
            # For this example, we'll just log the detection
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    @property
    def wake_word_detected(self) -> bool:
        """Check if a wake word was recently detected."""
        return self._wake_word_detected

    def reset_detection(self):
        """Reset the wake word detection flag."""
        self._wake_word_detected = False


async def main():
    """Main function demonstrating Porcupine wake word detection."""
    
    # Check for required access key
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        logger.error("Please set PICOVOICE_ACCESS_KEY environment variable")
        logger.error("Get your access key from https://console.picovoice.ai/")
        sys.exit(1)

    async with configure():
        # Configure transport
        transport = DailyTransport(
            room_url=os.getenv("DAILY_SAMPLE_ROOM_URL"),
            token=os.getenv("DAILY_SAMPLE_TOKEN"),
            bot_name="Porcupine Wake Word Bot",
            params=DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        # Create Porcupine filter with built-in keywords
        porcupine_filter = PorcupineFilter(
            access_key=access_key,
            keywords=["picovoice", "hey google", "alexa"],  # Built-in keywords
            sensitivities=[0.5, 0.7, 0.6],  # Adjust sensitivity per keyword
        )

        # Alternative: Use custom keyword files
        # porcupine_filter = PorcupineFilter(
        #     access_key=access_key,
        #     keyword_paths=["/path/to/custom_keyword.ppn"],
        #     sensitivities=[0.5],
        # )

        # Set the audio filter on the transport
        transport.set_audio_filter(porcupine_filter)

        # Create wake word handler
        wake_word_handler = WakeWordHandler()

        # Create LLM service (optional - for responding to wake words)
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )

        # Create context aggregator
        context = OpenAILLMContext()

        # Create pipeline
        pipeline = Pipeline([
            transport.input(),   # Audio input
            wake_word_handler,   # Handle wake word detections
            context.user(),      # User context
            llm,                 # LLM processing
            context.assistant(), # Assistant context
            transport.output(),  # Audio output
        ])

        # Create and run the task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        # Set up the runner
        runner = PipelineRunner()

        # Initial system message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. You have been activated by a wake word. "
                          "Respond briefly and helpfully to user requests. Keep responses concise.",
            }
        ]

        await task.queue_frames([context.get_context_frame(messages)])

        logger.info("🎙️  Porcupine wake word detection started!")
        logger.info(f"📢 Listening for wake words: {porcupine_filter.keywords}")
        logger.info("💡 Say one of the wake words to activate the assistant")

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
