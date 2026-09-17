import unittest

from tests import _env  # noqa: F401
import nl_descriptions as nd


class CleanDescriptionTest(unittest.TestCase):
    def test_strips_thinking_and_markdown(self):
        self.assertEqual(nd.clean_description("<think>\nlet me see\n</think>\n**Summary:** Loads a user.\nMore."),
                         "Loads a user.")
        self.assertEqual(nd.clean_description("- Handles login.\n- Second bullet."), "Handles login.")
        self.assertEqual(nd.clean_description('"Parses config."'), "Parses config.")
        self.assertEqual(nd.clean_description("```\nEncodes JWT.\n```"), "Encodes JWT.")

    def test_unterminated_thinking_and_empty(self):
        self.assertEqual(nd.clean_description("<thought>still thinking"), "")
        self.assertEqual(nd.clean_description(""), "")
        self.assertEqual(nd.clean_description("   \n  "), "")

    def test_length_cap(self):
        long = "Does a thing. " * 40
        out = nd.clean_description(long)
        self.assertLessEqual(len(out), nd.MAX_DESCRIPTION_CHARS + 3)
        self.assertTrue(out.endswith(".") or out.endswith("..."))

    def test_disabled_by_env(self):
        self.assertFalse(nd.is_enabled())


if __name__ == "__main__":
    unittest.main()
