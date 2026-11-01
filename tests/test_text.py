import unittest

from mmcqa_exp import prompts, settings as S
from mmcqa_exp.postprocess import clean, finalize, trim_unfinished


class PostprocessTests(unittest.TestCase):
    def test_markdown_turns_and_tokens_removed(self):
        raw = "**The Burj Khalifa** is tall.\n- It is in Dubai.\nHuman: And what else?\nAssistant: more"
        out = clean(raw)
        self.assertNotIn("**", out)
        self.assertNotIn("Human", out)
        self.assertNotIn("Assistant", out)
        self.assertIn("Burj Khalifa", out)
        self.assertEqual(clean("thought <channel|>The answer.<|im_end|>"), "The answer.")
        self.assertEqual(clean("<think>hidden</think>Visible."), "Visible.")
        self.assertEqual(clean("Answer: It is a dish."), "It is a dish.")
        self.assertEqual(clean("الإجابة: هذا طبق"), "هذا طبق")

    def test_never_empty(self):
        for raw in ("", None, "   ", "<think>only</think>", "**"):
            ans, fb = finalize(raw, "What is this?")
            self.assertTrue(ans.strip())
            self.assertTrue(fb)
        self.assertEqual(finalize("Fine.", "Q?"), ("Fine.", False))

    def test_trim_only_when_capped_and_keeps_single_sentence(self):
        text = "It is a mosque. It was built in the"
        self.assertEqual(trim_unfinished(text), "It is a mosque.")
        self.assertEqual(finalize(text, "Q", hit_cap=True)[0], "It is a mosque.")
        self.assertEqual(finalize(text, "Q", hit_cap=False)[0], text)
        self.assertEqual(finalize("It was built in the", "Q", hit_cap=True)[0], "It was built in the")
        self.assertEqual(trim_unfinished("جامع كبير. بني في"), "جامع كبير.")

    def test_arabic_survives(self):
        s = "برج خليفة هو أطول مبنى في العالم."
        self.assertEqual(clean(s), s)


class PromptTests(unittest.TestCase):
    def test_templates_contain_question_and_variety(self):
        for track, name in S.VARIETY.items():
            self.assertIn("Q1?", prompts.p_new("Q1?", track))
            self.assertIn(name, prompts.p_new("Q1?", track))
            self.assertIn(name, prompts.p_fusion("Q1?", track, "a", "b"))
            self.assertIn("HINT", prompts.p_cascade("Q1?", track, "HINT"))
        self.assertTrue(prompts.p0("Q?", "en").endswith("based on the image."))

    def test_fusion_order_deterministic_and_balanced(self):
        ids = [f"id{i}" for i in range(400)]
        first = [prompts.fusion_order(i) for i in ids]
        self.assertEqual(first, [prompts.fusion_order(i) for i in ids])
        self.assertTrue(150 < sum(first) < 250)

    def test_hint_is_capped(self):
        long = " ".join(["w"] * 500)
        self.assertLessEqual(len(prompts.p_cascade("Q", "en", long).split()), 80 + 40)


if __name__ == "__main__":
    unittest.main()
