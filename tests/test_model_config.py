import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import _env  # noqa: F401
import model_config as mc


class ModelConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models"
        self.hf = Path(self.tmp.name) / "hf"
        self.models.mkdir()
        self.hf.mkdir()
        self._patches = [
            mock.patch.object(mc, "MODELS_DIR", self.models),
            mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.hf)}, clear=False),
        ]
        for p in self._patches:
            p.start()
        for var in ("EMBED_MODEL_PATH", "CODE_RAG_EMBED_MODEL", "CODE_RAG_PROFILE",
                    "CODE_RAG_DESCRIPTION_MODEL", "CODE_RAG_DESCRIPTION_MODEL_KEY"):
            os.environ.pop(var, None)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.environ["EMBED_MODEL_PATH"] = str(_env.FAKE_MODEL_DIR)
        self.tmp.cleanup()

    def _download(self, key):
        d = self.models / mc.EMBED_MODELS[key]["local_dir"]
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"")

    def _cache(self, key):
        hf_id = mc.DESCRIPTION_MODELS[key]["hf_id"]
        (self.hf / f"models--{hf_id.replace('/', '--')}").mkdir(parents=True)

    def test_fresh_machine_uses_default_profile(self):
        self.assertEqual(mc.resolve_embed_model_key(), mc.PROFILES[mc.DEFAULT_PROFILE]["embed"])
        self.assertEqual(mc.resolve_description_model_key(), mc.PROFILES[mc.DEFAULT_PROFILE]["description"])

    def test_auto_detect_prefers_best_downloaded(self):
        self._download("sfr-embed-code-2b")
        self.assertEqual(mc.resolve_embed_model_key(), "sfr-embed-code-2b")
        self._download("qwen3-embed-0.6b")
        self.assertEqual(mc.resolve_embed_model_key(), "qwen3-embed-0.6b")
        self._cache("gemma-3-4b")
        self.assertEqual(mc.resolve_description_model_key(), "gemma-3-4b")
        self._cache("gemma-4-e4b")
        self.assertEqual(mc.resolve_description_model_key(), "gemma-4-e4b")

    def test_profile_beats_auto_detect(self):
        self._download("sfr-embed-code-2b")
        os.environ["CODE_RAG_PROFILE"] = "medium"
        self.assertEqual(mc.resolve_embed_model_key(), "qwen3-embed-0.6b")
        self.assertEqual(mc.resolve_description_model_key(), "gemma-4-e4b")
        os.environ["CODE_RAG_PROFILE"] = "bogus"
        with self.assertRaises(ValueError):
            mc.resolve_embed_model_key()

    def test_key_beats_profile_and_path_beats_key(self):
        os.environ["CODE_RAG_PROFILE"] = "high"
        os.environ["CODE_RAG_EMBED_MODEL"] = "qodo-embed-1.5b"
        self.assertEqual(mc.resolve_embed_model_key(), "qodo-embed-1.5b")
        self.assertTrue(mc.resolve_embed_model_path().endswith("qodo-embed-1-1.5b-mlx-q8"))
        os.environ["EMBED_MODEL_PATH"] = "/somewhere/else"
        self.assertIsNone(mc.resolve_embed_model_key())
        self.assertEqual(mc.resolve_embed_model_path(), "/somewhere/else")
        os.environ["CODE_RAG_DESCRIPTION_MODEL"] = "org/custom"
        self.assertEqual(mc.resolve_description_model_id(), "org/custom")

    def test_query_instruction_by_model_type(self):
        d = Path(self.tmp.name) / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        self.assertTrue(mc.query_instruction_for_model(str(d)).startswith("Instruct:"))
        self.assertTrue(mc.query_instruction_for_model(str(d)).endswith("Query:"))
        (d / "config.json").write_text(json.dumps({"model_type": "codexembed2b"}))
        self.assertIn("Instruct:", mc.query_instruction_for_model(str(d)))
        (d / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        self.assertEqual(mc.query_instruction_for_model(str(d)), "")
        self.assertEqual(mc.query_instruction_for_model("/nonexistent"), "")

    def test_summary_reports_downloaded_state(self):
        s = mc.summary()
        self.assertFalse(s["embed_model_downloaded"])
        self._download(mc.PROFILES[mc.DEFAULT_PROFILE]["embed"])
        self.assertTrue(mc.summary()["embed_model_downloaded"])


if __name__ == "__main__":
    unittest.main()
