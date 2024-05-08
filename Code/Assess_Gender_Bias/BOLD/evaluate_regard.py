import argparse
from datasets import load_dataset
import evaluate 
from random import sample
import json

from random import sample
from transformers import pipeline, AutoTokenizer

regard1 = evaluate.load("../../../Models/regard")
regard2 = evaluate.load("../../../Models/regard")

def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--input_file1', type=str, help='Input JSON file 1 containing prompts')
parser.add_argument('--input_file2', type=str, help='Input JSON file 2 containing prompts')
parser.add_argument('--output_path', type=str, help='Output path for saving results')
args = parser.parse_args()

input_data1 = load_from_json(args.input_file1)
input_data2 = load_from_json(args.input_file2)

round1_continuations = [list(entry.values())[0][0] for entry in input_data1]
round2_continuations = [list(entry.values())[0][0] for entry in input_data2]


regard1_del = regard1.compute(data = round1_continuations, aggregation = None)  
regard2_del = regard2.compute(data = round2_continuations, aggregation = None)  
print("computing response")

regard1_avg = regard1.compute(data=round1_continuations, aggregation='average')  
regard2_avg = regard2.compute(data=round2_continuations, aggregation='average')  
print("computing average")

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

save_to_json(regard1_del, f'{args.output_path}/regard_actor_del.json')
save_to_json(regard2_del, f'{args.output_path}/regard_actress_del.json')
save_to_json(regard1_avg, f'{args.output_path}/regard_actor_avg.json')
save_to_json(regard2_avg, f'{args.output_path}/regard_actress_avg.json')
print("results saved")