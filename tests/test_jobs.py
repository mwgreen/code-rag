import threading
import time
import unittest

from tests import _env  # noqa: F401
import jobs


class JobsTest(unittest.TestCase):
    def test_lifecycle_and_single_writer_per_project(self):
        gate = threading.Event()

        def slow(job):
            job.progress["step"] = 1
            gate.wait(5)
            if job.cancel.is_set():
                return {"cancelled": True}
            return {"ok": True}

        job = jobs.start_job("index", "/proj", slow)
        time.sleep(0.05)
        self.assertEqual(job.status, "running")
        self.assertIs(jobs.start_job("reconcile", "/proj", slow), job)   # same project: existing job returned
        other = jobs.start_job("index", "/other", lambda j: {"ok": True})
        self.assertIsNot(other, job)
        self.assertIs(jobs.active_job("/proj"), job)
        self.assertTrue(jobs.cancel_job(job.id))
        gate.set()
        for _ in range(100):
            if not job.active:
                break
            time.sleep(0.02)
        self.assertEqual(job.status, "cancelled")
        self.assertIsNone(jobs.active_job("/proj"))
        self.assertFalse(jobs.cancel_job(job.id))
        self.assertEqual(job.to_dict()["progress"], {"step": 1})

    def test_failure_is_captured(self):
        import logging
        logging.getLogger("code-rag.jobs").setLevel(logging.CRITICAL)  # expected failure below
        self.addCleanup(logging.getLogger("code-rag.jobs").setLevel, logging.NOTSET)

        def boom(job):
            raise RuntimeError("nope")
        job = jobs.start_job("index", "/fail", boom)
        for _ in range(100):
            if not job.active:
                break
            time.sleep(0.02)
        self.assertEqual(job.status, "failed")
        self.assertIn("RuntimeError: nope", job.error)
        self.assertIs(jobs.get_job(job.id), job)
        self.assertIn(job, jobs.list_jobs("/fail"))


if __name__ == "__main__":
    unittest.main()
