import tempfile
import unittest
from pathlib import Path

from tests import _env  # noqa: F401
from fts_hybrid import FTSIndex, build_match_query, rrf_merge


class MatchQueryTest(unittest.TestCase):
    def test_operators_and_punctuation_are_neutralized(self):
        q = build_match_query('does NOT use the cache for config.yaml user-authentication')
        self.assertNotIn(" NOT ", q.replace('"NOT"', ''))
        for tok in ('"does"', '"NOT"', '"cache"', '"config"', '"yaml"', '"user"', '"authentication"'):
            self.assertIn(tok, q)
        self.assertTrue(all(part.strip().startswith('"') for part in q.split(" OR ")))

    def test_dedup_short_tokens_and_empty(self):
        self.assertEqual(build_match_query("a b c"), "")
        self.assertEqual(build_match_query("Foo foo FOO"), '"Foo"')
        self.assertEqual(build_match_query("!!! ???"), "")


class FTSIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / ".code-rag" / "milvus.db")
        self.fts = FTSIndex("chunks_fts", ["path", "type"])
        conn = self.fts.connection(self.db)
        self.fts.insert(conn, [
            {"doc_id": "a::0", "content": "public User findUserById(long id) { return repo.get(id); }", "path": "/p/A.java", "type": "code"},
            {"doc_id": "b::0", "content": "Authentication uses JWT tokens signed with RS256", "path": "/p/B.md", "type": "documentation"},
            {"doc_id": "c::0", "content": "cache eviction policy: least recently used", "path": "/p/C.java", "type": "code"},
        ])
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_search_survives_punctuation_and_operators(self):
        hits = self.fts.search("how does jwt-authentication work? NOT tokens", n=5, db_path=self.db)
        self.assertEqual(hits[0]["doc_id"], "b::0")

    def test_filters(self):
        hits = self.fts.search("user id cache", n=5, filters={"type": "code"}, db_path=self.db)
        self.assertEqual({h["doc_id"] for h in hits}, {"a::0", "c::0"})
        hits = self.fts.search("user id cache", n=5, filters={"type": "documentation"}, db_path=self.db)
        self.assertEqual(hits, [])

    def test_insert_skips_duplicate_doc_ids(self):
        conn = self.fts.connection(self.db)
        self.fts.insert(conn, [{"doc_id": "a::0", "content": "dup", "path": "/p/A.java", "type": "code"}])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE doc_id='a::0'").fetchone()[0], 1)
        conn.close()

    def test_snapshot_and_delete(self):
        snap = self.fts.snapshot(self.db)
        self.assertEqual(snap["/p/A.java"], {"a::0"})
        self.assertEqual(set(snap), {"/p/A.java", "/p/B.md", "/p/C.java"})
        conn = self.fts.connection(self.db)
        self.fts.delete_doc_ids(conn, ["a::0", "c::0"])
        conn.close()
        self.assertEqual(set(self.fts.snapshot(self.db)), {"/p/B.md"})

    def test_schema_migration_recreates_table(self):
        other = FTSIndex("chunks_fts", ["path", "type", "extra"])
        conn = other.connection(self.db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks_fts)")]
        self.assertIn("extra", cols)
        conn.close()


class RRFTest(unittest.TestCase):
    def test_docs_in_both_lists_rank_first(self):
        vec = [{"doc_id": "x", "similarity": 0.9}, {"doc_id": "y", "similarity": 0.8}, {"doc_id": "z", "similarity": 0.7}]
        fts = [{"doc_id": "z"}, {"doc_id": "q"}]
        merged = rrf_merge(vec, fts, n=10)
        self.assertEqual(merged[0]["doc_id"], "z")
        self.assertEqual(merged[0]["similarity"], 0.7)   # vector copy wins
        self.assertEqual([m["doc_id"] for m in merged][1:], ["x", "y", "q"])


if __name__ == "__main__":
    unittest.main()
