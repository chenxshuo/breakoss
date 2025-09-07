
from breakoss.utils import free_gpu_memory
from breakoss.models import load_llm, InferenceConfig
from breakoss.evaluators import StrongREJECTEvaluator, HarmBenchEvaluator
from loguru import logger

def test_free_gpu_memory():
    model = load_llm(model_name="openai/gpt-oss-20b", cuda_device="auto", provider="hf")
    inference_config = InferenceConfig(
        max_new_tokens=1000,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=False,
    )
    response = model.chat(
        prompt="Hello, how are you?", inference_config=inference_config
    )

    free_gpu_memory(model)
    logger.info(f"Freed GPU memory after model inference.")
    import time
    time.sleep(5)

    strong_reject_evaluator = StrongREJECTEvaluator()
    strong_reject_evaluator.evaluate_one(
        harmful_prompt="Hello How are you?",
        response=response,
    )

    free_gpu_memory(strong_reject_evaluator)
    logger.info(f"Freed GPU memory after StrongREJECT evaluation.")
    import time
    time.sleep(5)

    harm_bench_evaluator = HarmBenchEvaluator()
    harm_bench_evaluator.evaluate_one(
        harmful_prompt="Hello How are you?",
        response=response,
    )
    free_gpu_memory(harm_bench_evaluator)

    logger.info(f"Freed GPU memory after HarmBench evaluation.")

    import time
    time.sleep(5)

if __name__ == "__main__":
    test_free_gpu_memory()