from breakoss.models import load_llm, InferenceConfig

model = load_llm(model_name="openai/gpt-oss-20b", cuda_device="auto", provider="hf")

inference_config = InferenceConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.0,
    do_sample=True
)

print(model.chat(prompt="Hello, how are you?", inference_config=inference_config))

