SUPPORTED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "microsoft/Phi-4-reasoning-plus",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",

    # "RealSafe/RealSafe-R1-7B"
]

SUPPORTED_DATASETS = [
    "StrongReject",
    "HarmfulBehaviors",
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