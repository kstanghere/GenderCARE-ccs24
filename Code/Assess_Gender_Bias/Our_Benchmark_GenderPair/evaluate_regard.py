import os
import json
import evaluate
import sys

# Check if sufficient arguments have been provided
if len(sys.argv) < 3:
    print("Usage: python evaluate_regard.py <file_path> <log_path>")
    sys.exit(1)

regard_model = evaluate.load("../../../Models/regard")

file_path = sys.argv[1]
print("data loaded!")
with open(file_path, 'r') as file:
    data = json.load(file)
    responses = [resp for item in data for resp in item.values()]
    regard_ratio = regard_model.compute(data = responses, aggregation = "average")
    log_path = sys.argv[2]
    with open(log_path, 'w') as log_file:
        log_file.write(f"Regard Score: {regard_ratio['average_regard']}\n")



