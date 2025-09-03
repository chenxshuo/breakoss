import json
import fire

def check_file(file_name: str="harmfulbehavior-1-structural-cot-bypass-results.json"):
    with open(file_name, "r") as json_file:
        results = json.load(json_file)

    if file_name != "harmfulprompts-fake-overrefusal-cotbypass-results.json":

        for i, result in enumerate(results):
            print("*"*100)
            print(f">>>>>>> Harmful Prompt: {result['datasample']['harmful_prompt']}")
            print(f">>>>>>>Generated Response: {result['evaluate_result']['response']}")
            print("*"*100)
            if i > 5:
                print("Omit the rest ...")
                break
    else:
        for i, result in enumerate(results):
            print("*"*100)
            print(f">>>>>>> Harmful Prompt: {result['prompt']}")
            print(f">>>>>>>Generated Response: {result['response']}")
            print("*"*100)
            if i > 5:
                print("Omit the rest ...")
                break

if __name__ == "__main__":
    fire.Fire(check_file)