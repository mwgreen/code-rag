import unittest

from tests import _env  # noqa: F401
import codechunk_wrapper as cw

JAVA = """package demo;

import java.util.List;

public class Greeter {
    private final List<String> names;

    public Greeter(List<String> names) { this.names = names; }

    public String greet(String who) {
        return "Hello, " + who + " from " + names.size() + " friends";
    }

    public int count() { return names.size(); }
}
"""


@unittest.skipUnless(cw.available(), "node / code-chunk not available")
class CodeChunkWrapperTest(unittest.TestCase):
    def test_shared_process_chunks_and_reports_status(self):
        chunks = cw.chunk_with_codechunk(JAVA, "/demo/Greeter.java", "java", max_size=400)
        self.assertTrue(chunks)
        self.assertTrue(all("content" in c and "start_line" in c for c in chunks))
        self.assertTrue(chunks[0]["content"].startswith("#"), "code-chunk context header expected")
        st = cw.status()
        self.assertTrue(st["available"])
        self.assertTrue(st["process_alive"])
        self.assertGreaterEqual(st["requests"], 1)
        # second call reuses the same process
        again = cw.chunk_with_codechunk(JAVA, "/demo/Greeter.java", "java", max_size=400)
        self.assertEqual(len(again), len(chunks))
        self.assertEqual(cw.status()["requests"], st["requests"] + 1)

    def test_unsupported_language_returns_none(self):
        self.assertIsNone(cw.chunk_with_codechunk("a: 1", "/x.yaml", "yaml"))


if __name__ == "__main__":
    unittest.main()
