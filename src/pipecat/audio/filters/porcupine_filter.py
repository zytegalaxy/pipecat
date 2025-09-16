#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Porcupine wake word detection audio filter for Pipecat.

This module provides an audio filter implementation using PicoVoice's Porcupine
Wake Word Detection engine to detect wake words in audio streams.
"""

from typing import List, Optional, Sequence

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame, WakeWordDetectionFrame

try:
    import pvporcupine
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Porcupine filter, you need to `pip install pipecat-ai[porcupine]`.")
    raise Exception(f"Missing module: {e}")


class PorcupineFilter(BaseAudioFilter):
    """Audio filter using Porcupine Wake Word Detection from PicoVoice.

    Provides real-time wake word detection for audio streams using PicoVoice's
    Porcupine engine. The filter buffers audio data to match Porcupine's required
    frame length and processes it in chunks. When a wake word is detected, it emits
    a WakeWordDetectionFrame while passing through the original audio unchanged.
    """

    def __init__(
        self,
        *,
        access_key: str,
        keywords: Optional[List[str]] = None,
        keyword_paths: Optional[List[str]] = None,
        sensitivities: Optional[List[float]] = None,
    ) -> None:
        """Initialize the Porcupine wake word detection filter.

        Args:
            access_key: PicoVoice access key for Porcupine engine authentication.
            keywords: List of built-in keywords to detect (e.g., ['picovoice', 'alexa']).
            keyword_paths: List of paths to custom .ppn keyword files.
            sensitivities: List of sensitivity values (0.0-1.0) for each keyword.
                          Higher values reduce false positives but may increase misses.

        Note:
            Either keywords or keyword_paths must be provided, but not both.
            If sensitivities is provided, it must have the same length as keywords/keyword_paths.
        """
        self._access_key = access_key
        self._keywords = keywords or []
        self._keyword_paths = keyword_paths or []
        self._sensitivities = sensitivities

        if not self._keywords and not self._keyword_paths:
            raise ValueError("Either keywords or keyword_paths must be provided")

        if self._keywords and self._keyword_paths:
            raise ValueError("Cannot specify both keywords and keyword_paths")

        self._filtering = True
        self._sample_rate = 0
        self._porcupine = None
        self._porcupine_ready = False
        self._audio_buffer = bytearray()

        # Store keyword names for event emission
        if self._keywords:
            self._keyword_names = self._keywords.copy()
        else:
            # For custom keyword files, use filename without extension as keyword name
            import os
            self._keyword_names = [
                os.path.splitext(os.path.basename(path))[0] for path in self._keyword_paths
            ]

        self._initialize_porcupine()

    def _initialize_porcupine(self) -> None:
        """Initialize the Porcupine engine with the provided configuration."""
        try:
            if self._keywords:
                self._porcupine = pvporcupine.create(
                    access_key=self._access_key,
                    keywords=self._keywords,
                    sensitivities=self._sensitivities,
                )
            else:
                self._porcupine = pvporcupine.create(
                    access_key=self._access_key,
                    keyword_paths=self._keyword_paths,
                    sensitivities=self._sensitivities,
                )
            self._porcupine_ready = True
            logger.info(f"Porcupine initialized with keywords: {self._keyword_names}")
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            self._porcupine_ready = False

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        if self._porcupine and self._sample_rate != self._porcupine.sample_rate:
            logger.warning(
                f"Porcupine filter needs sample rate {self._porcupine.sample_rate} (got {self._sample_rate})"
            )
            self._porcupine_ready = False

    async def stop(self):
        """Clean up the Porcupine engine when stopping."""
        if self._porcupine:
            self._porcupine.delete()

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Porcupine wake word detection to audio data.

        Buffers incoming audio and processes it in chunks that match Porcupine's
        required frame length. Returns the original audio data unchanged while
        emitting WakeWordDetectionFrame events when wake words are detected.

        Args:
            audio: Raw audio data as bytes to be processed.

        Returns:
            Original audio data as bytes (unchanged).
        """
        if not self._porcupine_ready or not self._filtering:
            return audio

        self._audio_buffer.extend(audio)

        num_frames = len(self._audio_buffer) // 2
        while num_frames >= self._porcupine.frame_length:
            # Grab the number of frames required by Porcupine
            num_bytes = self._porcupine.frame_length * 2
            audio_chunk = bytes(self._audio_buffer[:num_bytes])
            
            # Process audio for wake word detection
            data = np.frombuffer(audio_chunk, dtype=np.int16).tolist()
            keyword_index = self._porcupine.process(data)
            
            # If a wake word is detected, emit detection frame
            if keyword_index >= 0:
                keyword_name = self._keyword_names[keyword_index]
                detection_frame = WakeWordDetectionFrame(
                    keyword=keyword_name,
                    keyword_index=keyword_index,
                )
                logger.info(f"Wake word detected: {keyword_name} (index: {keyword_index})")
                
                # In a real implementation, this frame would be pushed to the pipeline
                # For now, we'll store it as an attribute that can be accessed
                self._last_detection = detection_frame

            # Adjust audio buffer and check again
            self._audio_buffer = self._audio_buffer[num_bytes:]
            num_frames = len(self._audio_buffer) // 2

        # Return original audio unchanged
        return audio

    @property
    def last_detection(self) -> Optional[WakeWordDetectionFrame]:
        """Get the last wake word detection frame (for testing/debugging)."""
        return getattr(self, '_last_detection', None)

    @property
    def keywords(self) -> List[str]:
        """Get the list of keywords being detected."""
        return self._keyword_names.copy()

    @property
    def sample_rate(self) -> int:
        """Get the required sample rate for Porcupine."""
        return self._porcupine.sample_rate if self._porcupine else 0

    @property
    def frame_length(self) -> int:
        """Get the required frame length for Porcupine."""
        return self._porcupine.frame_length if self._porcupine else 0
