"""End-to-end tests against real Milvus Lite + FTS5 with the fake embedder.

Covers: indexing, hybrid search score semantics, dedup, incremental hashing,
stale removal, verify/reconcile drift repair without re-embedding, concurrent
writers, clear.
"""

import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from tests import _env
from tests._env import write

rag_milvus = _env.install_fake_embedder()
import fts_hybrid  # noqa: E402

JAVA_AUTH = """package app.security;

import java.util.Optional;

public class JwtAuthenticationService {
    private final TokenVerifier verifier;

    public JwtAuthenticationService(TokenVerifier verifier) { this.verifier = verifier; }

    public Optional<UserPrincipal> authenticate(String bearerToken) {
        if (bearerToken == null || !bearerToken.startsWith("Bearer ")) return Optional.empty();
        return verifier.verifyJwt(bearerToken.substring(7)).map(UserPrincipal::fromClaims);
    }
}
"""

JAVA_CACHE = """package app.cache;

public class LruCacheEvictionPolicy {
    private final int capacity;
    public LruCacheEvictionPolicy(int capacity) { this.capacity = capacity; }
    public boolean shouldEvict(int size) { return size > capacity; }
    public String name() { return "least recently used eviction"; }
}
"""

DOC = """# Deployment guide

The service is deployed with docker compose. Set the DATABASE_URL and run migrations
before starting the containers. Health checks hit /health every 30 seconds.
"""


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rag_milvus.init_server_mode()

    @classmethod
    def tearDownClass(cls):
        rag_milvus.close_server_mode()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="code-rag-test-")
        self.root = str(Path(self.tmp).resolve())
        self.db = str(Path(self.root) / ".code-rag" / "milvus.db")
        write(Path(self.root) / "src/app/security/JwtAuthenticationService.java", JAVA_AUTH)
        write(Path(self.root) / "src/app/cache/LruCacheEvictionPolicy.java", JAVA_CACHE)
        write(Path(self.root) / "docs/deploy.md", DOC)

    def tearDown(self):
        rag_milvus._evict_client(self.db)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers --

    def index(self, **kw):
        return rag_milvus.index_directory(self.root, db_path=self.db, **kw)

    def fts_conn(self):
        return sqlite3.connect(fts_hybrid.FTSIndex.db_path(self.db))

    # -- tests --

    def test_index_search_and_score_semantics(self):
        stats = self.index()
        self.assertEqual(stats["files_indexed"], 3)
        self.assertEqual(stats["errors"], 0)
        self.assertGreaterEqual(stats["chunks_created"], 3)

        results = rag_milvus.search("jwt bearer token authentication", n=5, db_path=self.db)
        self.assertTrue(results)
        top = results[0]
        self.assertTrue(top["path"].endswith("JwtAuthenticationService.java"))
        self.assertIn(top["source"], ("semantic", "both", "keyword"))
        sims = [r["similarity"] for r in results if r["similarity"] is not None]
        self.assertTrue(sims, "semantic hits must carry a similarity")
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in sims))
        # Similarity is a similarity: the best match has the highest value, not the lowest.
        best = max(results, key=lambda r: r["similarity"] if r["similarity"] is not None else -2)
        self.assertTrue(best["path"].endswith("JwtAuthenticationService.java"))
        for r in results:
            self.assertNotIn("doc_id", r)
            self.assertNotIn("_rrf_score", r)
            self.assertIsInstance(r["start_line"], int)

        docs = rag_milvus.search("docker compose deployment", n=3, type_filter="documentation", db_path=self.db)
        self.assertTrue(docs and docs[0]["path"].endswith("deploy.md"))
        self.assertTrue(all(r["type"] == "documentation" for r in docs))

        java_only = rag_milvus.search("eviction", n=5, language_filter="java", db_path=self.db)
        self.assertTrue(all(r["language"] == "java" for r in java_only))

    def test_incremental_hashing_and_stale_removal(self):
        self.index()
        self.assertFalse(rag_milvus.file_needs_indexing(f"{self.root}/docs/deploy.md", self.db))
        self.assertEqual(self.index()["files_indexed"], 0)

        write(Path(self.root) / "docs/deploy.md", DOC + "\nKubernetes manifests live in deploy/k8s.\n")
        self.assertTrue(rag_milvus.file_needs_indexing(f"{self.root}/docs/deploy.md", self.db))
        stats = self.index()
        self.assertEqual(stats["files_indexed"], 1)

        os.remove(f"{self.root}/src/app/cache/LruCacheEvictionPolicy.java")
        stats = self.index()
        self.assertEqual(stats["files_removed"], 1)
        listed = rag_milvus.list_indexed_files(self.db)
        all_paths = {p for paths in listed.values() for p in paths}
        self.assertEqual(len(all_paths), 2)
        self.assertFalse(any(p.endswith("LruCacheEvictionPolicy.java") for p in all_paths))
        self.assertEqual(set(rag_milvus._fts.snapshot(self.db)), all_paths)

        stats = rag_milvus.get_stats(self.db)
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(sum(stats["by_type"].values()), stats["total_chunks"])

    def test_duplicate_files_dedupe_in_results(self):
        write(Path(self.root) / "src/copy/JwtAuthenticationService.java", JAVA_AUTH)
        self.index()
        results = rag_milvus.search("jwt bearer token authentication", n=10, db_path=self.db)
        auth_hits = [r for r in results if r["path"].endswith("JwtAuthenticationService.java")]
        self.assertGreaterEqual(len(auth_hits), 1)
        self.assertEqual(len({rag_milvus._content_fingerprint(r["content"]) for r in results}), len(results))

    def test_verify_and_reconcile_repair_drift_without_reembedding(self):
        self.index()
        self.assertTrue(rag_milvus.verify_index(self.root, self.db)["consistent"])

        # Drift 1: keyword rows vanish for one file (e.g. a crash between the two writes)
        auth = f"{self.root}/src/app/security/JwtAuthenticationService.java"
        conn = self.fts_conn()
        conn.execute("DELETE FROM chunks_fts WHERE path = ?", (auth,))
        conn.commit()
        conn.close()
        # Drift 2: an orphan keyword row for a file that was never in Milvus
        conn = self.fts_conn()
        conn.execute("INSERT INTO chunks_fts (doc_id, content, path, language, type, start_line, end_line, "
                     "class_name, component, description) VALUES ('ghost::0','ghost','/elsewhere/Ghost.java',"
                     "'java','code','1','2','','','')")
        conn.commit()
        conn.close()
        # Drift 3: a file deleted from disk, and a new file the watcher never saw
        os.remove(f"{self.root}/src/app/cache/LruCacheEvictionPolicy.java")
        write(Path(self.root) / "docs/runbook.md", "# Runbook\n\nRestart the worker with systemctl.\n")

        report = rag_milvus.verify_index(self.root, self.db)
        self.assertFalse(report["consistent"])
        self.assertEqual(report["fts_mismatch"], 1)
        self.assertEqual(report["fts_orphans"], 1)
        self.assertEqual(report["missing_on_disk"], 1)
        self.assertEqual(report["not_indexed"], 1)

        before = dict(_env.EMBED_CALLS)
        stats = rag_milvus.reconcile(self.root, self.db)
        self.assertEqual(stats["fts_rebuilt"], 1)
        self.assertEqual(stats["fts_orphans_removed"], 1)
        self.assertEqual(stats["removed_missing"], 1)
        self.assertEqual(stats["files_indexed"], 1)
        # Only the new runbook was embedded; the FTS rebuild came from Milvus rows.
        self.assertEqual(_env.EMBED_CALLS["texts"] - before["texts"], 1)
        self.assertTrue(rag_milvus.verify_index(self.root, self.db)["consistent"])
        hits = rag_milvus.search("bearer token", n=3, db_path=self.db)
        self.assertTrue(any(r["path"] == auth for r in hits))

    def test_excluded_files_are_removed_by_reconcile(self):
        self.index()
        write(Path(self.root) / ".ragconfig", "exclude_patterns:\n  - 'docs/*'\n")
        stats = rag_milvus.reconcile(self.root, self.db)
        self.assertEqual(stats["removed_missing"], 1)
        listed = rag_milvus.list_indexed_files(self.db)
        self.assertNotIn("documentation", listed)

    def test_concurrent_writers_do_not_corrupt(self):
        for i in range(9):
            write(Path(self.root) / f"src/gen/Svc{i}.java",
                  f"package gen;\npublic class Svc{i} {{ public int v() {{ return {i}; }} }}\n")
        errors = []
        from indexing_rules import IndexRules
        paths = sorted(IndexRules(self.root).walk())
        self.assertEqual(len(paths), 12)

        def worker(subset):
            try:
                for p in subset:
                    rag_milvus.add_file(p, force=True, db_path=self.db)
                    rag_milvus.search("service value", n=3, db_path=self.db)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(paths[i::3],)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        stats = rag_milvus.get_stats(self.db)
        self.assertEqual(stats["total_files"], len(paths))
        report = rag_milvus.verify_index(self.root, self.db)
        self.assertTrue(report["consistent"], report)

    def test_clear_collection(self):
        self.index()
        rag_milvus.clear_collection(self.db)
        self.assertEqual(rag_milvus.get_stats(self.db)["total_chunks"], 0)
        self.assertFalse(Path(fts_hybrid.FTSIndex.db_path(self.db)).exists())
        self.assertEqual(self.index()["files_indexed"], 3)

    def test_deterministic_primary_keys_and_filter_quoting(self):
        self.assertEqual(rag_milvus._doc_pk("/a/b.java::0"), rag_milvus._doc_pk("/a/b.java::0"))
        self.assertNotEqual(rag_milvus._doc_pk("/a/b.java::0"), rag_milvus._doc_pk("/a/b.java::1"))
        odd = Path(self.root) / 'src/we"ird/Na"me.java'
        write(odd, "package weird;\npublic class Name { public int x() { return 1; } }\n")
        self.assertGreater(rag_milvus.add_file(str(odd), db_path=self.db), 0)
        self.assertFalse(rag_milvus.file_needs_indexing(str(odd), self.db))
        self.assertGreater(rag_milvus.delete_by_path(str(odd), self.db), 0)


if __name__ == "__main__":
    unittest.main()
