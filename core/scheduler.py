# core/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.interval import IntervalTrigger
from typing import Callable, Any, Optional
import logging

# Configure logging for APScheduler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Job store keeps everything in memory (auto-cleared on shutdown)
jobstores = {
    'default': MemoryJobStore()
}

# Use a thread pool executor to run jobs in parallel threads
executors = {
    'default': ThreadPoolExecutor(max_workers=5)
}

# Default job settings
job_defaults = {
    'coalesce': False,       # do not coalesce missed runs
    'max_instances': 1       # only one instance of each job at a time
}


class Scheduler:
    """
    Wrapper around APScheduler BackgroundScheduler.
    Use add_interval_job() to schedule repeating tasks.
    """

    def __init__(self):
        self._sched = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        self._sched.start()
        logger.info("Scheduler started (in background)")

    def add_interval_job(
        self,
        func: Callable[..., Any],
        seconds: int,
        args: Optional[list[Any]] = None,
        job_id: Optional[str] = None
    ) -> None:
        """
        Schedule `func(*args)` to run every `seconds` seconds.
        If job_id is provided, you can later remove or modify this job.
        """
        trigger = IntervalTrigger(seconds=seconds)
        self._sched.add_job(
            func,
            trigger,
            args=args or [],
            id=job_id,
            replace_existing=True
        )
        logger.info(f"Added interval job '{job_id}' every {seconds}s")

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job by its ID."""
        self._sched.remove_job(job_id)
        logger.info(f"Removed job '{job_id}'")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler. `wait=False` stops immediately."""
        self._sched.shutdown(wait=wait)
        logger.info("Scheduler shut down")
