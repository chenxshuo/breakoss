<!--
<p align="center">
  <img src="https://github.com//breakoss/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  BreakOSS: Jailbreak GPT-OSS Toolkit
</h1>

<p align="center">
    <a href="https://github.com//breakoss/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com//breakoss/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/breakoss">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/breakoss" /></a>
    <a href="https://pypi.org/project/breakoss">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/breakoss" /></a>
    <a href="https://github.com//breakoss/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/breakoss" /></a>
    <a href='https://breakoss.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/breakoss/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="https://codecov.io/gh//breakoss/branch/main">
        <img src="https://codecov.io/gh//breakoss/branch/main/graph/badge.svg" alt="Codecov status" /></a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /></a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href="https://github.com//breakoss/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/></a>
    <!-- uncomment if you archive on zenodo
    <a href="https://zenodo.org/badge/latestdoi/XXXXXX">
        <img src="https://zenodo.org/badge/XXXXXX.svg" alt="DOI"></a>
    -->
</p>

A research toolkit providing a bag of techniques to jailbreak OpenAI’s
open-source models.

## 💪 Getting Started

### 🚀 Installation


```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 📦  Starting Point

- run the first method `Structural CoT Bypass`
```bash
python examples/1_cot_bypass.py one_example
```

- run the second method `Fake Over-Refusal`
```bash
python examples/2_fake_overrefusal.py one_example
```

- run the third method `Coercive Optimization`
```bash
python example/3_gcg.py
python example/3-1_gcg_transfer.py
```

- run the fourth method `Intent Hijack`
```bash
python examples/4_intent_hijack.py one_example
```

- run the fifth method `Plan Injection`

```bash
python examples/5_plan_injection.py one_example
```

## 🧑‍💻 More Scripts

- Evaluate Structural CoT Bypass on StrongReject or HarmfulBehaviors 
```bash
 python examples/1_cot_bypass.py main --dataset_name=StrongReject
 python examples/1_cot_bypass.py main --dataset_name=HarmfulBehaviors
```

- Evaluate Intent Hijack on StrongReject  
```bash
 python examples/4_intent_hijack.py eval_on_strongreject
```

- Evaluate Plan Injection on StrongReject  
```bash
 python examples/5_plan_injection.py eval_on_strongreject
```

## 📂 Folder Structure 

├── demo # submission to Kaggle 
│  ├── breakoss-findings-1-cotbypss.json
│  ├── breakoss-findings-2-fake-over-refusal-cotbypass.json
│  ├── breakoss-findings-3-coercive-optimization.json
│  ├── breakoss-findings-4-intent-hijack.json
│  ├── breakoss-findings-5-plan-injection.json
│  ├── demo_notebook.ipynb
│  └── results
├── docs
│  └── source
├── examples # example scripts to run each method
│  ├── 1_cot_bypass.py
│  ├── 2_fake_overrefusal.py
│  ├── 3-1_gcg_transfer.py
│  ├── 3_gcg.py
│  ├── 4_intent_hijack.py
│  ├── 5_plan_injection.py
│  ├── deploy.sh
│  ├── display.py
│  ├── llm_inference.py
│  ├── prepare_intent_plan.py
│  └── strongreject_with_intent_and_plan.json
├── pyproject.toml
├── src
│  └── breakoss
│      ├── api.py
│      ├── cli.py
│      ├── datasets # dataset loaders
│      ├── evaluators # evaluation implementations
│      ├── go_jailbreak.py
│      ├── __init__.py
│      ├── __main__.py
│      ├── methods # method implementations
│      ├── models # model wrappers
│      ├── __pycache__
│      ├── py.typed
│      ├── utils.py
│      └── version.py
│  └── breakoss
├── tests # tests for the package
├── tox.ini
└── uv.lock


### ⚖️ License

The code in this package is licensed under the MIT License.