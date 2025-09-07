SUPPORTED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "microsoft/phi-4-reasoning",
    "qwen/qwen3-4b-thinking-2507",
    "RealSafe/RealSafe-R1-7B"
]

SUPPORTED_DATASETS = [
    "HarmfulBehaviors",
    "StrongReject",
    "HarmBench",
    "CatQA",
    "JBBHarmfulBehaviors"
]


SUPPORTED_METRICS = [
    "RefusalWordsEvaluator",
    "LlamaGuardEvaluator",
    "StrongREJECTEvaluator",
    "HarmBenchEvaluator"
]