import tempfile
import unittest
from pathlib import Path

from tests import _env  # noqa: F401
import chunking
import rag_milvus


class ChunkingTest(unittest.TestCase):
    def test_chunk_default_splits_large_and_keeps_small(self):
        small = chunking.chunk_default("line\n" * 10, "/p/x.properties")
        self.assertEqual(len(small), 1)
        self.assertEqual((small[0]["start_line"], small[0]["end_line"]), (1, 11))
        big = chunking.chunk_default("\n".join(f"key{i}=value{i}" for i in range(2000)), "/p/x.properties")
        self.assertGreater(len(big), 1)
        self.assertEqual(big[0]["start_line"], 1)
        self.assertEqual(big[-1]["end_line"], 2000)
        for a, b in zip(big, big[1:]):
            self.assertEqual(b["start_line"], a["end_line"] + 1)

    def test_invalid_utf8_is_replaced_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Legacy.properties"
            p.write_bytes(b"caf\xe9=1\n" + b"key=value\n" * 20)
            chunks = chunking.chunk_file(str(p))
            self.assertEqual(len(chunks), 1)
            self.assertIn("caf�=1", chunks[0]["content"])

    def test_empty_path_and_empty_file(self):
        self.assertEqual(chunking.chunk_file(""), [])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.md"
            p.write_text("   \n")
            self.assertEqual(chunking.chunk_file(str(p)), [])

    def test_java_regex_fallback_has_metadata(self):
        src = "package a;\n\npublic class Foo {\n" + "    public void m() { int x = 1; }\n" * 20 + "}\n"
        chunks = chunking.chunk_java(src, "/p/Foo.java")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["language"], "java")
        self.assertEqual(chunks[0]["type"], "code")
        self.assertEqual(chunks[0]["class_name"], "Foo")

    def test_fingerprint_ignores_context_header(self):
        a = "# a/b/File.java\n# Scope: X\n# Uses: Y\n\nint f() { return 1; }"
        b = "# other/File.java\n# Scope: Q\n\nint  f()  {  return 1; }"
        c = "# a/b/File.java\n# Scope: X\n\nint g() { return 2; }"
        self.assertEqual(rag_milvus._content_fingerprint(a), rag_milvus._content_fingerprint(b))
        self.assertNotEqual(rag_milvus._content_fingerprint(a), rag_milvus._content_fingerprint(c))

    def test_detect_type(self):
        self.assertEqual(chunking.detect_type("x.md", "markdown"), "documentation")
        self.assertEqual(chunking.detect_type("x.json", "json"), "config")
        self.assertEqual(chunking.detect_type("x.proto", "protobuf"), "code")


if __name__ == "__main__":
    unittest.main()
