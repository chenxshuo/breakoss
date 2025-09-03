
- `breakoss-toobox.zip` includes our toolbox for this Kaggle competition.
- `breakoss-deep-research.zip` includes our implementation of Deep Research Agent powered by gpt-oss-20b.
- `DeepResearchReport.md` shows the generated research report from Deep Research Agent powered by gpt-oss-20b. It shows a comprehensive and detailed content about the controlled medicine amphetamines.
- `check_json.py` is a helper script to check these extra submitted json files (not for the 5 Kaggle datasets).
  ```bash
  # uv pip install fire # if you don't have fire installed
  python check_json.py --file_name=harmfulbehavior-1-structural-cot-bypass-results.json
  ```
- `harmfulbehavior-1-structural-cot-bypass-results.json` is the full results of Structural CoT Bypass method on the first 150 questions of HarmfulBehavior dataset.
- `harmfulbehavior-2-structural-cot-bypass-results.json` is the full results of Structural CoT Bypass method on the last 150 questions of HarmfulBehavior dataset.
- `harmfulprompts-fake-overrefusal-cotbypass-results.json` is the full results of Fake Over-Refusal + CoT Bypass method on failure cases on both harmfulBehaviors and StrongReject Datasets.
- `strongreject-structural-cot-bypass-results.json` is the full results of Structural CoT Bypass method on StrongReject dataset.
- `strongreject-1-intent-hijack-results.json` is the full results of Intent Hijack method on the first 150 questions of StrongReject dataset.
- `strongreject-2-intent-hijack-results.json` is the full results of Intent Hijack method on the last 150 questions of StrongReject dataset.
- `strongreject-1-plan-injection-results.json` is the full results of Plan Injection method on the first 150 questions of StrongReject dataset.
- `strongreject-2-plan-injection-results.json` is the full results of Plan Injection method on the last 150 questions of StrongReject dataset.