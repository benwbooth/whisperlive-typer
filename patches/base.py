import json
import logging
import threading
import time
import queue
import numpy as np


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    min_avg_logprob: float
    """Segments with avg_logprob below this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.35,
        clip_audio=False,
        same_output_threshold=4,
        translation_queue=None,
        min_avg_logprob=-0.8,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.min_avg_logprob = min_avg_logprob
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue

        # Segment tracking for duplicate detection
        self.last_segment_end = 0.0  # Track end time to detect gaps and duplicates

        # For sending finalized text to client
        self.finalized_to_send = None  # Set by update_segments, used by send_transcription_to_client

        # threading
        self.lock = threading.Lock()

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    # If we had pending text and VAD now sees silence, explicitly clear it.
                    if self.current_out:
                        self.current_out = ""
                        self.prev_out = ""
                        self.same_output_count = 0
                        self.end_time_for_same_output = None
                        self.send_transcription_to_client(self.prepare_segments())
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError

    def format_segment(self, start, end, text, completed=False, no_speech_prob=0.0, avg_logprob=0.0):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            completed (bool): Whether this segment is finalized.
            no_speech_prob (float): Probability that segment contains no speech (for filtering hallucinations).
            avg_logprob (float): Average log probability (confidence measure).

        Returns:
            dict: A dictionary representing the formatted transcription segment.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed,
            'no_speech_prob': no_speech_prob,
            'avg_logprob': avg_logprob,
        }

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.lock.release()

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends transcription update to the client over the websocket connection.

        Simple protocol:
        - finalize: text that is now final (client should type and never delete)
        - text: current pending text (client can update via diff)

        Args:
            segments: List of segment dicts for backward compatibility
        """
        msg = {"uid": self.client_uid}

        # Include finalized text if any was set by update_segments
        if self.finalized_to_send:
            msg["finalize"] = self.finalized_to_send
            self.finalized_to_send = None  # Clear after sending

        # Always include pending text, even when empty, so clients can clear stale pending.
        msg["text"] = self.current_out

        # Keep segments for backward compatibility / debugging
        msg["segments"] = segments

        try:
            self.websocket.send(json.dumps(msg))
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True

    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_avg_logprob(self, segment):
        return getattr(segment, "avg_logprob", 0)

    def segment_passes_filters(self, segment):
        return (
            self.get_segment_no_speech_prob(segment) <= self.no_speech_thresh
            and self.get_segment_avg_logprob(segment) >= self.min_avg_logprob
        )

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    @staticmethod
    def join_finalized_texts(chunks):
        merged = ""
        for chunk in chunks:
            if not chunk:
                continue
            if (
                merged
                and merged[-1].isascii() and merged[-1].isalnum()
                and chunk[0].isascii() and chunk[0].isalnum()
            ):
                merged += " "
            merged += chunk
        return merged

    def update_segments(self, segments, duration):
        """
        Processes the segments from Whisper and updates the transcript.

        Simplified protocol:
        - When segments complete, send {finalize: text} for each completed segment
        - Always send {text: pending} with current pending text

        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.

        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None
        finalized_texts = []  # Collect all finalized texts to send

        if not segments:
            self.current_out = ""
            self.prev_out = ""
            self.same_output_count = 0
            self.end_time_for_same_output = None
            return None

        logging.info(f"update_segments: len={len(segments)}, duration={duration:.2f}, timestamp_offset={self.timestamp_offset:.2f}")
        for i, s in enumerate(segments):
            logging.info(f"  seg[{i}]: '{s.text}' start={self.get_segment_start(s):.2f} end={self.get_segment_end(s):.2f} no_speech={self.get_segment_no_speech_prob(s):.2f}")

        # Process completed segments independently from the last pending segment.
        if len(segments) > 1:
            for s in segments[:-1]:
                text_ = s.text
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + min(duration, self.get_segment_end(s))
                if start >= end:
                    continue
                if not self.segment_passes_filters(s):
                    logging.debug(
                        "  Dropping low-confidence completed segment: "
                        f"no_speech={self.get_segment_no_speech_prob(s):.2f}, "
                        f"avg_logprob={self.get_segment_avg_logprob(s):.2f}"
                    )
                    continue

                # Skip segments we've already processed (prevents duplicate finalization)
                if self.last_segment_end > 0 and start < self.last_segment_end - 0.1:
                    logging.debug(f"  Skipping already-processed segment: start={start:.2f} < last_end={self.last_segment_end:.2f}")
                    continue

                # This segment is newly completed - add to finalized list
                self.text.append(text_)
                finalized_texts.append(text_)

                completed_segment = self.format_segment(
                    start, end, text_, completed=True,
                    no_speech_prob=self.get_segment_no_speech_prob(s),
                    avg_logprob=self.get_segment_avg_logprob(s),
                )
                self.transcript.append(completed_segment)
                self.last_segment_end = end

                if self.translation_queue:
                    try:
                        self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning("Translation queue is full, skipping segment")
                offset = min(duration, self.get_segment_end(s))
                logging.info(f"  Completed segment: '{text_}' end={end:.2f}, offset={offset:.2f}")

        # Process the last segment as pending if confidence is acceptable.
        if self.segment_passes_filters(segments[-1]):
            with self.lock:
                pending_start = self.timestamp_offset + self.get_segment_start(segments[-1])
                pending_end = self.timestamp_offset + min(duration, self.get_segment_end(segments[-1]))

            self.current_out = segments[-1].text
            with self.lock:
                last_segment = self.format_segment(
                    pending_start, pending_end,
                    self.current_out,
                    completed=False,
                    no_speech_prob=self.get_segment_no_speech_prob(segments[-1]),
                    avg_logprob=self.get_segment_avg_logprob(segments[-1]),
                )
        else:
            # Low-confidence pending segment - clear pending.
            with self.lock:
                pending_start = self.timestamp_offset + self.get_segment_start(segments[-1])
                pending_end = self.timestamp_offset + min(duration, self.get_segment_end(segments[-1]))
                last_segment = self.format_segment(
                    pending_start, pending_end,
                    "",
                    completed=False,
                    no_speech_prob=self.get_segment_no_speech_prob(segments[-1]),
                    avg_logprob=self.get_segment_avg_logprob(segments[-1]),
                )
            self.current_out = ""
            logging.debug(
                "  Clearing pending due to low confidence: "
                f"no_speech={self.get_segment_no_speech_prob(segments[-1]):.2f} "
                f"avg_logprob={self.get_segment_avg_logprob(segments[-1]):.2f}"
            )

        # Handle repeated output logic (stall detection)
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times, finalize it
        if self.same_output_count > self.same_output_threshold:
            last_no_speech = self.get_segment_no_speech_prob(segments[-1])
            last_avg_logprob = self.get_segment_avg_logprob(segments[-1])

            # Don't finalize repeated low-confidence output (likely hallucination).
            if not self.segment_passes_filters(segments[-1]):
                logging.info(
                    f"  Skipping repeated low-confidence segment: '{self.current_out}' "
                    f"no_speech_prob={last_no_speech:.2f} avg_logprob={last_avg_logprob:.2f}"
                )
                offset = min(duration, self.end_time_for_same_output)
            elif not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                # Finalize the repeated pending segment
                self.text.append(self.current_out)
                finalized_texts.append(self.current_out)

                with self.lock:
                    seg_start = self.timestamp_offset
                    seg_end = self.timestamp_offset + min(duration, self.end_time_for_same_output)
                    completed_segment = self.format_segment(
                        seg_start, seg_end, self.current_out, completed=True,
                        no_speech_prob=last_no_speech,
                        avg_logprob=self.get_segment_avg_logprob(segments[-1]),
                    )
                    self.transcript.append(completed_segment)
                    self.last_segment_end = seg_end

                    if self.translation_queue:
                        try:
                            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning("Translation queue is full, skipping segment")

                offset = min(duration, self.end_time_for_same_output)
                logging.info(f"  Finalized repeated segment: '{self.current_out}'")

            self.current_out = ''
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        # Advance timestamp_offset
        if offset is not None:
            with self.lock:
                old_offset = self.timestamp_offset
                self.timestamp_offset += offset
                logging.info(f"  Advanced timestamp_offset: {old_offset:.2f} + {offset:.2f} = {self.timestamp_offset:.2f}")
        elif len(segments) == 1:
            # Single segment - advance conservatively to prevent buffer overflow
            seg_start = self.get_segment_start(segments[0])
            if seg_start > 0.5:
                conservative_advance = seg_start - 0.5
                with self.lock:
                    old_offset = self.timestamp_offset
                    self.timestamp_offset += conservative_advance
                    logging.info(f"  Single-segment conservative advance: {old_offset:.2f} + {conservative_advance:.2f} = {self.timestamp_offset:.2f}")

        # Store finalized text for send_transcription_to_client to use
        if finalized_texts:
            self.finalized_to_send = self.join_finalized_texts(finalized_texts)
            logging.info(f"  Finalized text queued: '{self.finalized_to_send}'")

        return last_segment
