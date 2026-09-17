import os
import tempfile
import unittest
from pathlib import Path

from tests import _env  # noqa: F401
import tools


def hit(path, sim, source="semantic", **extra):
    d = {"path": path, "language": "java", "type": "code", "content": "code", "start_line": 1, "end_line": 2,
         "similarity": sim, "source": source}
    d.update(extra)
    return d


class RelevanceTest(unittest.TestCase):
    def test_floor_keeps_high_similarity_and_keyword_hits(self):
        results = [hit("/a", 0.82), hit("/b", 0.31), hit("/c", None, "keyword"), hit("/d", 0.10)]
        kept, below = tools._apply_relevance_floor(results, 0.30)
        self.assertFalse(below)
        self.assertEqual([r["path"] for r in kept], ["/a", "/b", "/c"])

    def test_floor_all_below_returns_everything_with_flag(self):
        kept, below = tools._apply_relevance_floor([hit("/a", 0.1), hit("/b", 0.2)], 0.5)
        self.assertTrue(below)
        self.assertEqual(len(kept), 2)

    def test_floor_disabled(self):
        results = [hit("/a", 0.01)]
        self.assertEqual(tools._apply_relevance_floor(results, 0.0), (results, False))

    def test_format_shows_similarity_not_its_inverse(self):
        text = tools.format_results([hit("/a", 0.87), hit("/b", None, "keyword"), hit("/c", 0.5, "both")])
        self.assertIn("**Relevance:** 0.87", text)
        self.assertNotIn("0.13", text)
        self.assertIn("**Match:** keyword", text)
        self.assertIn("**Match:** semantic+keyword", text)
        self.assertNotIn("Low semantic similarity", text)

    def test_low_similarity_note_only_when_best_is_low(self):
        self.assertIn("Low semantic similarity", tools.format_results([hit("/a", 0.2), hit("/b", 0.3)]))
        self.assertNotIn("Low semantic similarity", tools.format_results([hit("/a", 0.2), hit("/b", 0.75)]))

    def test_search_result_hints(self):
        text = tools._format_search_results([hit("/a", 0.9), hit("/a", 0.8), hit("/b", 0.7)], 0.0)
        self.assertIn("read_file", text)
        self.assertIn("/a (2 chunks)", text)


class RootScopingTest(unittest.TestCase):
    def test_inside_outside_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "src").mkdir()
            self.assertEqual(tools.resolve_under_root(str(root / "src"), str(root)), str(root / "src"))
            self.assertEqual(tools.resolve_under_root("src", str(root)), str(root / "src"))
            self.assertEqual(tools.resolve_under_root(str(root), str(root)), str(root))
            with self.assertRaises(ValueError):
                tools.resolve_under_root("/etc/passwd", str(root))
            with self.assertRaises(ValueError):
                tools.resolve_under_root(str(root / "src" / ".." / ".." / "x"), str(root))
            with self.assertRaises(ValueError):
                tools.resolve_under_root(str(root) + "2", str(root))


class FormattersTest(unittest.TestCase):
    def test_verify_and_job_formatting(self):
        report = {"consistent": False, "disk_files": 10, "indexed_files": 9, "indexed_chunks": 30, "fts_chunks": 28,
                  "missing_on_disk": 0, "not_indexed": 1, "changed": 0, "fts_orphans": 0, "fts_mismatch": 1,
                  "not_indexed_sample": ["/p/new.java"], "fts_mismatch_sample": ["/p/x.java"],
                  "missing_on_disk_sample": [], "changed_sample": [], "fts_orphans_sample": []}
        text = tools.format_verify(report)
        self.assertIn("**Consistent:** no", text)
        self.assertIn("/p/new.java", text)
        self.assertIn("repair=true", text)

        import jobs
        job = jobs.Job(id="abc", kind="index", project_root="/p")
        job.progress.update(files_done=3, files_total=10, current="X.java")
        text = tools.format_job(job, "Started.")
        self.assertIn("3/10", text)
        self.assertIn("index_status", text)
        self.assertEqual(tools.format_job(None), "No indexing job found for this project.")


if __name__ == "__main__":
    unittest.main()
