"""Bounded in-memory jobs with per-request inference cancellation."""
import threading
import time
import uuid


class Cancelled(Exception):
    pass


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.created = time.monotonic()
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.events = []
        self.done = False
        self.closer = None

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled('Cancelled by user')

    def publish(self, kind, **data):
        with self.lock:
            self.events.append({'type': kind, **data})

    def attach(self, closer):
        with self.lock:
            self.check()
            self.closer = closer

    def detach(self):
        with self.lock:
            self.closer = None

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            closer = self.closer
        if closer:
            try:
                closer()
            except OSError:
                pass

    def snapshot(self, cursor=0):
        with self.lock:
            return {'id': self.id, 'events': self.events[cursor:], 'cursor': len(self.events),
                    'done': self.done, 'cancelled': self.cancelled.is_set()}


class JobRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}

    def create(self):
        with self.lock:
            self.jobs = {key: job for key, job in self.jobs.items()
                         if not job.done or time.monotonic() - job.created < 900}
            if sum(not job.done for job in self.jobs.values()) >= 4:
                raise ValueError('Four requests are already running; stop one first.')
            job = Job()
            self.jobs[job.id] = job
            return job

    def get(self, key):
        with self.lock:
            return self.jobs.get(key)


JOBS = JobRegistry()
