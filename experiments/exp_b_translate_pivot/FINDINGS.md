# Experiment B findings: translate-pivot vs direct answering

Model: fine-tuned Qwen3.5-4B (S2). Items: first 100 dev ids of MSA, Egyptian, Levantine (300). Translator: Claude Sonnet 5.5
via OpenRouter (paid, behaviour study only). Judge: Gemini 2.5 Flash. Charts: `results/report/deep_dive/`.

## 1. Results

| System | Official macro F1 | Judge correct |
|---|---|---|
| Direct (answer in Arabic) | 0.7738 | 58.7% |
| Pivot, plain translator | 0.7604 | 71.3% |
| Pivot, original English-track question, plain translator | 0.7634 | 73.0% |
| **Pivot, style-matched translator (12 examples)** | **0.7757** | **70.0%** |
| Ceiling: gold English answer, plain translator | 0.8637 | - |
| Ceiling: gold English answer, style-matched translator | 0.8924 | 99.7% |

Style-matched pivot minus direct: F1 +0.002 (95% CI -0.005 to +0.008), judge +11.3 points (CI +5.7 to +17.3).
Per track (F1 / judge): MSA +0.017 / +11, Egyptian -0.014 / +10, Levantine +0.003 / +13.

## 2. What the diagnostics show

1. **The judge gain is real content, not translator fluency.** The pivot's English answers, judged in English against the
   English reference before any translation, are 70 / 67 / 72% correct (MSA / Egyptian / Levantine questions). After
   translation they are 71 / 68 / 75%. Direct is 61 / 57 / 58%. The model knows the answer more often than it can express it
   in Arabic: answering in Arabic costs about 11-13 points of correctness.
2. **The F1 loss of the plain pivot is wording, not content.** BERTScore precision drops 0.786 -> 0.760 while recall is
   unchanged (0.764 -> 0.762). On the 149 items both systems answer correctly, F1 still drops 0.790 -> 0.770 and ROUGE-L
   0.330 -> 0.280.
3. **The references are translations of the English reference** (all splits share ids and order across tracks). The
   direct model learned that translation style in fine-tuning; a generic translator writes differently and is penalised.
   Showing the translator 12 reference-style examples raises the gold-answer ceiling from 0.864 to 0.892 and the pivot from
   0.760 to 0.776.
4. **Question translation is not the bottleneck.** Using the original English-track question instead of the translated one
   moves F1 by +0.003 and the judge by +1.7 points.
5. **Egyptian is the hard case for style.** Its references are freer and more colloquial (style-matched ceiling 0.868 vs
   0.919 for MSA), and the style-matched pivot is still -0.014 F1 there.
6. Verdict changes vs direct (style-matched pivot): 68 items improve, 31 get worse; 149 are correct in both.

## 3. Recommendation

* **Adopt the pivot architecture for the Arabic tracks, but only with a reference-style translator.** It is the only
  setup that keeps official F1 (no loss) while gaining about 11 judge points.
* **Replace the paid translator with a local one trained on the task's own parallel data.** Train has 10,000 items with
  the same id in all four tracks, so there are 10,000 (English answer -> MSA / Egyptian / Levantine answer) pairs in the
  exact reference style. Fine-tune a small local model (for example a second LoRA on the same Qwen3.5-4B, or a dedicated
  MT model) on these pairs. Expected to beat the 12-example prompt, because the gold-answer ceiling shows style matters by
  about 0.03 F1.
* **Cheaper single-model alternative to test alongside:** fine-tune the answerer to write the English answer first and then
  its translation in one generation (targets from the same parallel data). One model, one call, no external translator.
* Keep direct as the fallback per track: if the local translator does not reach the direct F1 on Egyptian, use direct
  there and pivot for MSA and Levantine.
* Open question for the organisers: devtest ids are also shared across tracks, so the English-track question for the same
  id is available. Using it removes question translation, but check whether cross-track inputs are allowed before relying
  on it. About 2% of Egyptian questions in dev/devtest are actually answers (no question), which hurts any setup.

## 4. Caveats

* 100 items per track; per-track judge CIs are wide (about +-10 points).
* The judge is one model (Gemini). The ceiling check (99.7% correct for translated gold answers) shows it is not biased
  against translated Arabic.
* Few-shot examples come from dev ids 101-500, never from the evaluated 100.
* Translation cost for all of Experiment B including diagnostics: about $4.

## 5. Reproduce

```
python experiments/exp_b_translate_pivot/diagnostics.py                      # 4 diagnostic systems (cached translations)
$SC -m local_analysis.score --runs experiments/exp_b_translate_pivot/results/diagnostics experiments/exp_b_translate_pivot/results/assembled results/phase1/full --out experiments/exp_b_translate_pivot/results/diagnostics/analysis --systems sft_S2 pivot_s2 ref_tr ref_tr_fs pivot_fs entrack_tr --allow-partial
$PY -m local_analysis.judge --out experiments/exp_b_translate_pivot/results/diagnostics/analysis --systems sft_S2 pivot_s2 pivot_fs entrack_tr ref_tr_fs
$PY -m local_analysis.judge --out experiments/exp_b_translate_pivot/results/analysis --systems sft_S2 pivot_s2 pivot_en_msa pivot_en_arz pivot_en_ajp
python experiments/exp_b_translate_pivot/deep_dive.py                        # charts d1-d6 in results/report/deep_dive/
```
