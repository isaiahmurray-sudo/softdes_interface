"""Utilities for recording and replaying controller timing characteristics."""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass

TIME_LOG_LEVEL = 5
logging.addLevelName(TIME_LOG_LEVEL, "TIME")

@dataclass
class TimingRecord:
    """Single timing observation for a named controller operation."""

    id: str
    type: str
    duration: float
    start_time: float
    end_time: float = None

class TimingRecorder():
    """Capture operation durations and provide average delay replay support."""

    def __init__(self, filename: str | None = None):
        """Initialize recorder state and optionally load prior timing records."""
        self._records = {}
        self._filename = self._filename_build_path(filename) if filename else None
        self._delay_cache = {}
        if self._filename:
            self._load_records(self._filename)
    
    def _filename_build_path(self, filename : str) -> str:
        """Resolve timing filenames into the repository ``timing_data`` directory."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        timing_dir = os.path.join(repo_root, "softdes_interface/timing_data")
        os.makedirs(timing_dir, exist_ok=True)

        base_name = os.path.basename(filename)
        return os.path.join(timing_dir, base_name)
    
    def start_record(self, record_type: str) -> str:
        """Start a timing record and return its generated identifier."""
        record_id = str(uuid.uuid4())
        self._records[record_id] = TimingRecord(id=record_id, type=record_type, duration=None, start_time=time.time())
        return record_id
    
    def end_record(self, record_id: str):
        """Complete a record, computing and logging elapsed duration."""
        end_time = time.time()
        record = self._records.get(record_id)
        if record is None:
            logging.error(f"Attempted to end timing record with id {record_id} which does not exist.")
            return
        record.end_time = end_time
        record.duration = record.end_time - record.start_time
        logging.info(f"Ended timing record {record_id} of type '{record.type}' with duration {record.duration:.4f} seconds.")
        logging.log(TIME_LOG_LEVEL, f"Timing Record - ID: {record.id}, Type: {record.type}, Duration: {record.duration:.4f} seconds")
        return record.duration
    
    def _load_records(self, filename: str):
        """Load prior timing records from JSON into memory."""
        path = self._filename_build_path(filename)
        if not os.path.exists(path):
            logging.info(f"Timing record file does not exist yet: {path}")
            return

        logging.info(f"Loading timing records from {path}")
        with open(path, "r") as f:
            data = json.load(f)
        for record_data in data:
            record = TimingRecord(**record_data)
            self._records[record.id] = record
        logging.info(f"Loaded {len(self._records)} timing records from {path}")

    def save_records(self, filename: str | None = None):
        """Persist current in-memory timing records to JSON."""
        if filename is None:
            if self._filename is None:
                raise ValueError("No filename configured for TimingRecorder")
            path = self._filename
        else:
            path = self._filename_build_path(filename)
        logging.info(f"Saving timing records to {path}")
        with open(path, "w") as f:
            json.dump([record.__dict__ for record in self._records.values()], f, indent=4)
        logging.info(f"Saved {len(self._records)} timing records to {path}")
    
    def get_delay(self, record_type: str) -> float:
        """Return cached average delay for a record type, defaulting to 0."""
        if record_type in self._delay_cache:
            return self._delay_cache[record_type]
        else:
            logging.warning(f"Delay for record type '{record_type}' not found in cache. Returning 0.")
            return 0.0
    
    def delay(self, record_type: str) -> float:
        """Sleep for the cached delay associated with ``record_type``."""
        delay = self.get_delay(record_type)
        logging.log(TIME_LOG_LEVEL, f"Applying delay of {delay:.4f} seconds for record type '{record_type}'")
        time.sleep(delay)
        return delay

    def delete_record(self, record_id: str):
        """Remove a timing record by id if present."""
        if record_id in self._records:
            del self._records[record_id]
            logging.info(f"Deleted timing record with id {record_id}")
        else:
            logging.warning(f"Attempted to delete timing record with id {record_id} which does not exist.")

    def calculate_delay_cache(self):
        """Recompute per-type average durations used by ``delay`` replay."""
        sorted_types = {}
        for record in self._records.values():
            if record.type not in sorted_types:
                sorted_types[record.type] = []
            sorted_types[record.type].append(record.duration)

        for record_type, durations in sorted_types.items():
            valid_durations = [d for d in durations if d is not None]
            if valid_durations:
                avg_duration = sum(valid_durations) / len(valid_durations)
                self._delay_cache[record_type] = avg_duration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._filename:
            self.save_records(self._filename)
        return False