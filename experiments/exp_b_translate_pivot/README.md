# Experiment B: translate-pivot vs direct answering for MSA, Egyptian and Levantine

**Question.** The model is best in English. For the three non-English tracks, is it better to (a) answer in the language directly, or (b) translate the question to English, answer in English, and translate the answer back?

**Design.** Model: the fine-tuned Qwen3.5-4B (S2 adapter) for both ways. Items: the first 100 dev items of msa/arz/ajp (300 items).
* **Direct**: `sft_S2` from Phase 1 on the same items (the question and the answer both stay in the language).
* **Pivot**: question -> English (OpenRouter `anthropic/claude-sonnet-5.5`, temperature 0), the model answers in English with the training prompt, answer -> target variety (same translator, told to write natural colloquial Egyptian / Levantine or MSA). The translator sees only the text it translates (no image, no original question). It is a paid model for this behaviour study and is to be replaced by a local one later.
* Translation runs on the Mac (it needs the OpenRouter key; the key is never sent to Kaggle). The GPU step runs on Kaggle.
* Extra diagnostic: the English intermediate answers are scored against the **English** reference of the same item, to see how much is lost in translation versus in the model.

| Step | Where | Command |
|---|---|---|
| 1. Translate questions | local | `python experiments/exp_b_translate_pivot/translate_questions.py --rows 100` |
| 2. Upload inputs (adapter + questions) | local | `experiments/exp_b_translate_pivot/build_dataset.sh` |
| 3. Answer in English | Kaggle | `kaggle kernels push -p experiments/exp_b_translate_pivot/kaggle` |
| 4. Download | local | `kaggle kernels output muhammadhassan31/mmcqa-exp-b-translate-pivot -p experiments/exp_b_translate_pivot/results/kaggle`, unzip `exp_b_results.zip` there |
| 5. Translate answers back | local | `python experiments/exp_b_translate_pivot/translate_back.py` |
| 6. Official scores | local | `$SC -m local_analysis.score --runs experiments/exp_b_translate_pivot/results/assembled results/phase1/full --out experiments/exp_b_translate_pivot/results/analysis --systems sft_S2 pivot_s2 pivot_en_msa pivot_en_arz pivot_en_ajp --allow-partial` |
| 7. Judge | local | `$PY -m local_analysis.judge --out experiments/exp_b_translate_pivot/results/analysis --systems sft_S2 pivot_s2` |
| 8. Charts | local | `$PY experiments/exp_b_translate_pivot/analyze.py` -> `results/report/` |

Files: `translator.py`, `translate_questions.py`, `run_kaggle.py`, `translate_back.py`, `analyze.py`, `data/` (translations and cache), `results/` (gitignored).
