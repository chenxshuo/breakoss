"""
Extract pure plans from the PlanInjection
"""
import pandas as pd



def transform(
        ori_csv: str = "transformed_datasets/jbb-harmful-behaviors_PlanInjection.csv",
        key_name: str = "Goal",
        save_path: str = "transformed_datasets/jbb-harmful-behaviors_PlanInjection_pure_plan.csv"
):
    ds = pd.read_csv(ori_csv)

    for index, row in ds.iterrows():
        harmful_prompt = row[key_name]
        query = harmful_prompt.split("<|end|><|start|>assistant<|channel|>analysis<|message|>")[0]
        print("Query:", query)
        plan = harmful_prompt.split("<|end|><|start|>assistant<|channel|>analysis<|message|>")[1].split("<|end|><|start|>assistant<|channel|>commentary<|message|>")[0]
        print("Plan:", plan)
        ds.at[index, key_name] = query + "\n" + plan

    ds.to_csv(save_path, index=False)

if __name__ == "__main__":
    transform(
        ori_csv="transformed_datasets/strongreject_dataset_PlanInjection.csv",
        key_name="forbidden_prompt",
        save_path="transformed_datasets/strongreject_dataset_PlanInjection_pure_plan.csv"
    )

    transform(
        ori_csv="transformed_datasets/harmful_behaviors_PlanInjection.csv",
        key_name="goal",
        save_path="transformed_datasets/harmful_behaviors_PlanInjection_pure_plan.csv"
    )

    transform(
        ori_csv="transformed_datasets/harmbench_behaviors_text_all_PlanInjection.csv",
        key_name="Behavior",
        save_path="transformed_datasets/harmbench_behaviors_text_all_PlanInjection_pure_plan.csv"
    )
