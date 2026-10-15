"""Bedrock backend: batching, retry, truncation, Converse descriptions and the
index identity check, against a stubbed bedrock-runtime client.

The opt-in integration test (RUN_BEDROCK_IT=1) calls the real service with the
configured models: CODE_RAG_BEDROCK_EMBED_MODEL (default us.cohere.embed-v4:0)
and CODE_RAG_BEDROCK_DESC_MODEL (default us.amazon.nova-micro-v1:0).
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tests import _env

try:
    from botocore.exceptions import ClientError
    HAVE_BOTO = True
except ImportError:  # boto3 is only needed for the bedrock backend
    HAVE_BOTO = False

import bedrock_backend as bb


def client_error(code: str, op: str = "InvokeModel"):
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


def fake_vector(text: str, dim: int):
    return _env.fake_embed([text])[0][:dim] + [0.0] * max(0, dim - _env.DIM)


class StubBedrock:
    """Records calls; `failures` is a list of exceptions raised by the next calls, in order."""

    def __init__(self, failures=None, dims_override=None, converse_text="Adds two numbers."):
        self.invocations = []
        self.converse_calls = []
        self.failures = list(failures or [])
        self.dims_override = dims_override
        self.converse_text = converse_text
        self.lock = threading.Lock()

    def _maybe_fail(self):
        with self.lock:
            if self.failures:
                raise self.failures.pop(0)

    def invoke_model(self, modelId, body, **kwargs):
        req = json.loads(body)
        with self.lock:
            self.invocations.append((modelId, req))
        self._maybe_fail()
        if "texts" in req:  # Cohere
            dim = self.dims_override or req["output_dimension"]
            payload = {"embeddings": {"float": [fake_vector(t, dim) for t in req["texts"]]}}
            headers = {"x-amzn-bedrock-input-token-count": str(sum(len(t.split()) for t in req["texts"]))}
        else:  # Titan
            dim = self.dims_override or req["dimensions"]
            payload = {"embedding": fake_vector(req["inputText"], dim),
                       "inputTextTokenCount": len(req["inputText"].split())}
            headers = {}
        return {"body": io.BytesIO(json.dumps(payload).encode()),
                "ResponseMetadata": {"HTTPHeaders": headers}}

    def converse(self, modelId, messages, inferenceConfig):
        with self.lock:
            self.converse_calls.append((modelId, messages, inferenceConfig))
        self._maybe_fail()
        return {"output": {"message": {"content": [{"text": self.converse_text}]}},
                "usage": {"inputTokens": 40, "outputTokens": 8}}


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class BedrockTestCase(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        self._orig_sleep, self._orig_random = bb._sleep, bb._random
        bb._sleep = self.sleeps.append
        bb._random = lambda: 1.0  # deterministic: the upper bound of the jitter window
        bb.USAGE.reset()

    def tearDown(self):
        bb._sleep, bb._random = self._orig_sleep, self._orig_random


class CohereEmbedTests(BedrockTestCase):
    def test_batches_of_96_in_order_with_input_type(self):
        stub = StubBedrock()
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=256, client=stub)
        texts = [f"text number {i}" for i in range(200)]
        vectors = emb.embed(texts)
        self.assertEqual([len(req["texts"]) for _, req in stub.invocations], [96, 96, 8])
        self.assertTrue(all(req["input_type"] == "search_document" for _, req in stub.invocations))
        self.assertTrue(all(req["output_dimension"] == 256 for _, req in stub.invocations))
        self.assertEqual(vectors, [fake_vector(t, 256) for t in texts])
        usage = bb.USAGE.snapshot()["us.cohere.embed-v4:0"]
        self.assertEqual((usage["calls"], usage["texts"]), (3, 200))
        self.assertGreater(usage["input_tokens"], 0)

    def test_query_uses_search_query(self):
        stub = StubBedrock()
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", client=stub)
        vec = emb.embed(["where is auth"], input_type="search_query")[0]
        self.assertEqual(len(vec), 1024)
        self.assertEqual(stub.invocations[0][1]["input_type"], "search_query")

    def test_wrong_dimension_from_service_is_an_error(self):
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=512, client=StubBedrock(dims_override=256))
        with self.assertRaisesRegex(RuntimeError, "256 dims, expected 512"):
            emb.embed(["x"])

    def test_bare_profile_only_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "us.cohere.embed-v4:0"):
            bb.BedrockEmbedder("cohere.embed-v4:0", client=StubBedrock())

    def test_unsupported_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=768, client=StubBedrock())

    def test_long_text_is_truncated_with_a_warning(self):
        stub = StubBedrock()
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=256, client=stub, max_chars=100)
        with self.assertLogs("bedrock", level="WARNING") as logs:
            emb.embed(["a" * 500, "short"])
        self.assertEqual([len(t) for t in stub.invocations[0][1]["texts"]], [100, 5])
        self.assertIn("Truncated 1 of 2", "\n".join(logs.output))
        self.assertNotIn("aaaa", "\n".join(logs.output), "text must never be logged")
        self.assertEqual(bb.USAGE.snapshot()["us.cohere.embed-v4:0"]["truncated"], 1)


class TitanEmbedTests(BedrockTestCase):
    def test_one_text_per_call_concurrently_in_order(self):
        stub = StubBedrock()
        emb = bb.BedrockEmbedder("amazon.titan-embed-text-v2:0", client=stub, concurrency=8)
        texts = [f"chunk {i} body" for i in range(30)]
        vectors = emb.embed(texts)
        self.assertEqual(len(stub.invocations), 30)
        for _, req in stub.invocations:
            self.assertEqual((req["dimensions"], req["normalize"]), (1024, True))
            self.assertNotIn("texts", req)
        self.assertEqual(vectors, [fake_vector(t, 1024) for t in texts])
        self.assertEqual(bb.USAGE.snapshot()["amazon.titan-embed-text-v2:0"]["texts"], 30)


class RetryTests(BedrockTestCase):
    def test_retries_throttling_with_exponential_backoff(self):
        stub = StubBedrock(failures=[client_error("ThrottlingException"),
                                     client_error("ServiceUnavailableException"),
                                     client_error("ModelNotReadyException")])
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=256, client=stub)
        self.assertEqual(len(emb.embed(["x"])[0]), 256)
        self.assertEqual(len(stub.invocations), 4)
        self.assertEqual(self.sleeps, [0.5, 1.0, 2.0])
        self.assertEqual(bb.USAGE.snapshot()["us.cohere.embed-v4:0"]["retries"], 3)

    def test_backoff_is_capped_and_jittered(self):
        self.assertEqual(bb.backoff_delay(20), bb.BACKOFF_CAP)
        bb._random = lambda: 0.25
        self.assertEqual(bb.backoff_delay(3), 0.5)

    def test_non_retryable_error_raises_at_once(self):
        stub = StubBedrock(failures=[client_error("AccessDeniedException")])
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", client=stub)
        with self.assertRaises(ClientError):
            emb.embed(["x"])
        self.assertEqual((len(stub.invocations), self.sleeps), (1, []))

    def test_gives_up_after_max_attempts(self):
        stub = StubBedrock(failures=[client_error("ThrottlingException")] * 10)
        body = json.dumps({"inputText": "x", "dimensions": 256, "normalize": True})
        with self.assertRaises(ClientError):
            bb.call_with_retry(lambda: stub.invoke_model(modelId="m", body=body), "m", max_attempts=3)
        self.assertEqual(len(stub.invocations), 3)
        self.assertEqual(len(self.sleeps), 2)


class ResumeTests(BedrockTestCase):
    """After sleep/hibernation the pooled connections are dead: the client is replaced."""

    def setUp(self):
        super().setUp()
        self.clock = {"wall": 1000.0, "mono": 50.0}
        self.made = []
        self.saved = (bb._wall, bb._mono, bb.make_client)
        bb._wall = lambda: self.clock["wall"]
        bb._mono = lambda: self.clock["mono"]

        def factory(max_pool_connections=16):
            self.made.append(StubBedrock())
            return self.made[-1]
        bb.make_client = factory

    def tearDown(self):
        bb._wall, bb._mono, bb.make_client = self.saved
        super().tearDown()

    def advance(self, wall, mono):
        self.clock["wall"] += wall
        self.clock["mono"] += mono

    def test_client_replaced_after_suspend_only(self):
        emb = bb.BedrockEmbedder("us.cohere.embed-v4:0", dim=256)
        emb.embed(["a"])
        self.advance(wall=600, mono=600)          # idle but awake: keep the client
        emb.embed(["b"])
        self.assertEqual(len(self.made), 1)
        self.advance(wall=3600 * 12, mono=2)      # hibernated overnight
        with self.assertLogs("bedrock", level="INFO") as logs:
            emb.embed(["c"])
        self.assertEqual(len(self.made), 2)
        self.assertIn("reconnecting", "\n".join(logs.output))
        self.assertEqual(len(self.made[1].invocations), 1)

    def test_injected_client_is_never_replaced(self):
        stub = StubBedrock()
        gen = bb.BedrockGenerator("us.amazon.nova-micro-v1:0", client=stub)
        gen.generate("x")
        self.advance(wall=3600, mono=1)
        gen.generate("y")
        self.assertEqual((len(stub.converse_calls), self.made), (2, []))


class GeneratorTests(BedrockTestCase):
    def test_converse_request_and_usage(self):
        stub = StubBedrock()
        gen = bb.BedrockGenerator("us.amazon.nova-micro-v1:0", client=stub, max_tokens=80)
        self.assertEqual(gen.generate("Summarize"), "Adds two numbers.")
        model, messages, config = stub.converse_calls[0]
        self.assertEqual(model, "us.amazon.nova-micro-v1:0")
        self.assertEqual(config, {"maxTokens": 80, "temperature": 0.0})
        self.assertEqual(messages[0]["content"][0]["text"], "Summarize")
        usage = bb.USAGE.snapshot()["us.amazon.nova-micro-v1:0"]
        self.assertEqual((usage["calls"], usage["input_tokens"], usage["output_tokens"]), (1, 40, 8))


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class DescriptionTests(BedrockTestCase):
    """nl_descriptions on the bedrock backend (module state patched; the suite runs as mlx)."""

    def setUp(self):
        super().setUp()
        import nl_descriptions
        self.nd = nl_descriptions
        self.saved = (nl_descriptions.BACKEND, nl_descriptions.MODEL_ID, nl_descriptions._bedrock,
                      nl_descriptions._model_load_failed)
        self.stub = StubBedrock(converse_text="Summary: Parses the input file.\nMore text")
        nl_descriptions.BACKEND = "bedrock"
        nl_descriptions.MODEL_ID = "us.amazon.nova-micro-v1:0"
        nl_descriptions._bedrock = bb.BedrockGenerator(nl_descriptions.MODEL_ID, client=self.stub,
                                                       max_tokens=nl_descriptions.BEDROCK_MAX_TOKENS)
        nl_descriptions._model_load_failed = False
        self.tmp = Path(tempfile.mkdtemp(prefix="code-rag-desc-"))
        self.db = str(self.tmp / ".code-rag" / "milvus.db")
        self.chunks = [{"type": "code", "content": f"def f{i}(x):\n    return x + {i}\n" + "#" * 120}
                       for i in range(12)]

    def tearDown(self):
        (self.nd.BACKEND, self.nd.MODEL_ID, self.nd._bedrock, self.nd._model_load_failed) = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_describes_in_parallel_and_caches(self):
        out = self.nd.describe_chunks(self.chunks, db_path=self.db)
        self.assertEqual(out, ["Parses the input file."] * 12)
        self.assertEqual(len(self.stub.converse_calls), 12)
        _, messages, config = self.stub.converse_calls[0]
        self.assertEqual(config, {"maxTokens": 80, "temperature": 0.0})
        self.assertTrue(messages[0]["content"][0]["text"].startswith("Summarize this code in one sentence."))
        self.nd.describe_chunks(self.chunks, db_path=self.db)
        self.assertEqual(len(self.stub.converse_calls), 12, "second pass must come from the cache")

    def test_access_denied_disables_descriptions(self):
        self.stub.failures = [client_error("AccessDeniedException", "Converse")] * 12
        with self.assertLogs("code-rag.descriptions", level="WARNING"):
            out = self.nd.describe_chunks(self.chunks, db_path=self.db)
        self.assertEqual(out, [None] * 12)
        self.assertFalse(self.nd.is_enabled())


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class IndexIdentityTests(BedrockTestCase):
    """rag_milvus identity check and an end-to-end index/search on a stubbed Bedrock embedder."""

    COHERE = "us.cohere.embed-v4:0"
    TITAN = "amazon.titan-embed-text-v2:0"

    def setUp(self):
        super().setUp()
        import rag_milvus
        self.rm = rag_milvus
        self.saved = {k: getattr(rag_milvus, k) for k in
                      ("BACKEND", "_BEDROCK_MODEL", "_EMBED_DIM", "_bedrock_embedder", "embed_texts")}
        # Use the real embed_texts (the suite's fake embedder replaces it in other tests).
        rag_milvus.embed_texts = getattr(rag_milvus, "_real_embed_texts", rag_milvus.embed_texts)
        self.tmp = Path(tempfile.mkdtemp(prefix="code-rag-bedrock-"))
        self.db = str(self.tmp / ".code-rag" / "milvus.db")
        self.meta = self.tmp / ".code-rag" / "model_config.json"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(self.rm, k, v)
        self.rm._prepared_dbs.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def use_bedrock(self, model, dim, stub=None):
        self.rm.BACKEND = "bedrock"
        self.rm._BEDROCK_MODEL = model
        self.rm._EMBED_DIM = dim
        self.stub = stub or StubBedrock()
        self.rm._bedrock_embedder = bb.BedrockEmbedder(model, dim=dim, client=self.stub)

    def write_meta(self, data):
        self.meta.parent.mkdir(parents=True, exist_ok=True)
        self.meta.write_text(json.dumps(data))

    def test_mlx_index_refused_by_bedrock_backend(self):
        self.write_meta({"embed_model_path": "/Users/x/models/qwen3", "embed_dim": 1024})
        self.use_bedrock(self.COHERE, 1024)
        with self.assertRaises(self.rm.ModelMismatchError) as ctx:
            self.rm._check_model_consistency(self.db)
        msg = str(ctx.exception)
        self.assertIn("backend mlx vs bedrock", msg)
        self.assertIn("--clear", msg)

    def test_other_bedrock_model_same_dimension_refused(self):
        self.write_meta({"backend": "bedrock", "embed_model": self.TITAN, "embed_dim": 1024})
        self.use_bedrock(self.COHERE, 1024)
        with self.assertRaisesRegex(self.rm.ModelMismatchError, "model amazon.titan"):
            self.rm._check_model_consistency(self.db)

    def test_other_dimension_refused(self):
        self.write_meta({"backend": "bedrock", "embed_model": self.COHERE, "embed_dim": 1024})
        self.use_bedrock(self.COHERE, 512)
        with self.assertRaisesRegex(self.rm.ModelMismatchError, "dimension 1024 vs 512"):
            self.rm._check_model_consistency(self.db)

    def test_matching_identity_accepted_and_recorded(self):
        self.use_bedrock(self.COHERE, 256)
        self.rm._check_model_consistency(self.db)
        self.assertEqual(json.loads(self.meta.read_text()),
                         {"backend": "bedrock", "embed_model": self.COHERE, "embed_dim": 256})
        self.rm._check_model_consistency(self.db)  # second open: same identity, no error

    def test_legacy_mlx_identity_still_accepted_on_mlx(self):
        self.write_meta({"embed_model_path": "/old/path", "embed_dim": self.rm._get_embed_dim()})
        self.rm._check_model_consistency(self.db)
        self.assertEqual(json.loads(self.meta.read_text())["backend"], "mlx")

    def test_index_and_search_end_to_end_then_clear_recovers(self):
        self.use_bedrock(self.COHERE, 256)
        _env.write(self.tmp / "src" / "Auth.java",
                   "public class JwtUserResolution {\n  public User resolve(String token) {\n"
                   "    return users.findBySubject(parse(token).subject());\n  }\n}\n")
        stats = self.rm.index_directory(str(self.tmp), db_path=self.db)
        self.assertGreaterEqual(stats["chunks_created"], 1)
        hits = self.rm.search("JwtUserResolution resolve token", n=3, db_path=self.db)
        self.assertTrue(hits and hits[0]["path"].endswith("Auth.java"))
        types = [req["input_type"] for _, req in self.stub.invocations]
        self.assertIn("search_document", types)
        self.assertEqual(types[-1], "search_query")

        # Switching models: refused until the index is cleared.
        self.rm._prepared_dbs.clear()
        self.use_bedrock(self.TITAN, 256)
        with self.assertRaises(self.rm.ModelMismatchError):
            self.rm.search("token", db_path=self.db)
        self.rm.clear_collection(db_path=self.db)
        self.rm.index_directory(str(self.tmp), db_path=self.db)
        self.assertEqual(json.loads(self.meta.read_text())["embed_model"], self.TITAN)


class NoMlxImportTests(unittest.TestCase):
    def test_bedrock_backend_never_imports_mlx(self):
        code = ("import sys, http_server, rag_milvus, nl_descriptions\n"
                "rag_milvus.load_embedding_model()\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('mlx', 'mlx_embeddings', 'mlx_lm'))\n"
                "print(rag_milvus.BACKEND, nl_descriptions.MODEL_ID, bad)\n"
                "sys.exit(1 if bad else 0)\n")
        env = dict(os.environ, CODE_RAG_BACKEND="bedrock", PYTHONPATH=str(_env.ROOT),
                   CODE_RAG_BEDROCK_EMBED_MODEL="amazon.titan-embed-text-v2:0")
        env.pop("EMBED_MODEL_PATH", None)
        if not HAVE_BOTO:
            self.skipTest("boto3 not installed")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(_env.ROOT), env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr[-2000:])
        self.assertIn("bedrock us.amazon.nova-micro-v1:0 []", proc.stdout)


@unittest.skipUnless(os.getenv("RUN_BEDROCK_IT") == "1", "set RUN_BEDROCK_IT=1 to call Bedrock")
class BedrockIntegrationTest(unittest.TestCase):
    def test_real_embeddings_and_description(self):
        model = os.getenv("CODE_RAG_BEDROCK_EMBED_MODEL") or bb.DEFAULT_EMBED_MODEL
        emb = bb.BedrockEmbedder(model)
        docs = emb.embed(["public class WarehouseRefreshOrchestrator { void refresh() {} }",
                          "Chocolate cake recipe with butter and sugar"])
        query = emb.embed(["which class orchestrates warehouse refreshes"], input_type="search_query")[0]
        self.assertEqual({len(docs[0]), len(docs[1]), len(query)}, {emb.dim})

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))
        self.assertGreater(cos(query, docs[0]), cos(query, docs[1]))

        gen = bb.BedrockGenerator(os.getenv("CODE_RAG_BEDROCK_DESC_MODEL") or bb.DEFAULT_DESCRIPTION_MODEL)
        text = gen.generate("Summarize this code in one sentence. Be concise - describe WHAT it does, "
                            "not HOW.\n\n```\ndef add(a, b):\n    return a + b\n```\n\nOne-sentence summary:")
        self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()
