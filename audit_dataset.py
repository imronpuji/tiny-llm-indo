import json
import os

def check_counts(base_path):
    files = [f for f in os.listdir(base_path) if f.endswith(".json")]
    report = {}
    for f in files:
        file_path = os.path.join(base_path, f)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                report[f] = len(data)
        except Exception:
            report[f] = "Error"
    
    return report

base_dir = "/Users/muhamadimron/Projects/llm/dataset_topics/"
results = check_counts(base_dir)

print(json.dumps(results, indent=2))
