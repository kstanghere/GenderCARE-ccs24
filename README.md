# GenderCARE

This official repository contains the source code, datasets, and scripts for the paper "*[GenderCARE: A Comprehensive Framework for Assessing and Reducing Gender Bias in Large Language Models](https://arxiv.org/pdf/2408.12494)*," accepted by [ACM CCS 2024](https://www.sigsac.org/ccs/CCS2024/home.html), authored by Kunsheng Tang, Wenbo Zhou, and Jie Zhang et al. 

In this paper, we introduce Gender**CARE**, a comprehensive framework that provides innovative **C**riteria, **A**ssessment methods, **R**eduction techniques, and **E**valuation metrics for quantifying and mitigating gender bias in large language models (LLMs). Our approach aims to offer a realistic assessment and reduction of gender biases, hoping to make a significant step towards achieving fairness and equity in LLMs.

## Overview

Our artifact repository includes:
- GenderPair, our proposed benchmark for assessing gender bias in LLMs 
- Code for assessing gender bias in LLMs using our benchmark GenderPair, as well as existing benchmarks StereoSet, Winoqueer, and BOLD 
- Code and datasets for reducing gender bias using our proposed debiasing strategy
- Scripts to generate the key results tables and figures from the paper, including:
  - Perplexity probability difference on original vs. lightly modified prompts (EEC) 
  - Bias-Pair Ratio, Toxicity, and Regard on GenderPair benchmark
  - Results on StereoSet, Winoqueer, and BOLD after debiasing

## Getting Started

### Directory Structure

The key directories in this repository are:

- `Code/`: Contains Python scripts to assess bias, run debiasing, and generate tables/plots
  - `Assess_Gender_Bias/`: Scripts to evaluate bias on GenderPair, StereoSet, Winoqueer, and BOLD
  - `Reduce_Gender_Bias/`: Scripts to fine-tune models using our debiasing strategy
  - `EEC_Modify/`: Scripts to evaluate perplexity probability differences (ppd) on lightly modified prompts (Figure 1)
- `Datasets/`: Contains the bias evaluation datasets used in our experiments
  - `Assess_Gender_Bias/`: Subdirectories for GenderPair, StereoSet, Winoqueer, and BOLD
- `Models/`: Contains model checkpoints before and after debiasing
  - `base_models/`: Original pre-trained models, e.g., Meta Llama2-chat
  - `ft_models/`: Fine-tuned debiased versions of the models

### Environment Setup

Our code has been tested on Linux (a server with 4 x NVIDIA A6000 GPUs, each with 48GB memory) with Python 3.10.13, CUDA 11.7, PyTorch 2.0.1, and HuggingFace Transformers 4.31.0.

To set up the environment, follow these three steps:

1. Install CUDA 11.7, pytorch 2.0.1, python 3.10.13, numpy 1.26.4 and mpi4py 3.1.4 within a `conda` virtual environment.

2. Run the following command to install the other required packages listed in the `requirements.txt` file in the current directory:

   ```
   pip install -r requirements.txt
   ```

3. Run the following Python script to check if the GPU and CUDA environment are correctly recognized and available for use:

   ```python
   import torch
   
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```

   If `torch.cuda.is_available()` returns `True`, the environment is ready. 


### Pre-trained LLM Download

In this repository, we use Llama2-13B model as an example to reproduce the main results from the paper. There are two ways to obtain this pre-trained model:

1. Download directly from the official repository on Hugging Face:
   - Make sure you have git-lfs installed. If not, run the command: `git lfs install`
   - Execute the command `git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf` to download the model and place it in the `Models/base_models` directory.
   - Note that this method is suitable for users who have previously obtained access to the Meta Llama series. If you have never used it before, you need to fill out the Access Form on the [<u>meta-llama/Llama-2-13b-chat-hf</u>](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) page and submit a request. This process may take some time, so we recommend referring to the second method to obtain the model for these users.

2. Download from the backup on Hugging Face:
   - We have fully backed up all the files and pre-trained models from [<u>meta-llama/Llama-2-13b-chat-hf</u>](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) on Hugging Face at [<u>Warddamn2/Meta-Llama-2-13b-chat-hf</u>](https://huggingface.co/Warddamn2/Meta-Llama-2-13b-chat-hf). You can directly download the model without requesting access permission by running the command: `git clone https://huggingface.co/Warddamn2/Meta-Llama-2-13b-chat-hf`. Also, you should make sure you have git-lfs installed before git clone.

Regardless of which method you choose, you need to place the pre-trained model in the `Models/base_models` directory and rename the root directory of the pre-trained model to `Meta-Llama-2-13b-chat-hf` to ensure a smooth click-to-run experience.


If you want to use other models, you can modify the `--model_path`, `--model_type`, and other parameters in the scripts below. 
### Pre-trained Toxicity and Regard Models Download

1. Toxicity
   
   Navigate to the `Models/toxicity/facebook` directory and execute the command:  `git clone https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target`


2. Regard
   
   Navigate to the `Models/regard/sasha` directory and execute the command: `git clone https://huggingface.co/sasha/regardv3`



##  Assess Gender Bias in LLMs

### Perplexity Probability Difference on EEC 

Validates the claim that LLMs are more sensitive to meaning-preserving template modifications.

1. Navigate to the `Code/EEC_Modify/` directory
2. Run the evaluation script on Llama-2 13B:
   
```bash 
CUDA_VISIBLE_DEVICES=0 deepspeed \
  --master_port=60850 evaluate_ppd.py \
  --model_path ../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --output_file ppd_result_EEC_llama2_13b.csv \
  --data_path EEC_examples.csv \
  --zero_stage 3 \
  --model_type llama2 &> ppd_EEC_llama2_13b.log
```

3. The script will generate `ppd_result_EEC_llama2_13b.csv` containing the perplexity probability differences between original and lightly modified prompts that preserve meaning but switch genders.


### GenderPair Results on Original Models 

Validates the gender bias in off-the-shelf LLMs using our GenderPair benchmark in terms of bias-pair ratio, toxicity, and regard.

#### Step 1: Generate model responses 

1. Navigate to `Code/Assess_Gender_Bias/Our_Benchmark_GenderPair/`
2. For each gender group, run the `assess.py` script to get model responses:
   
- Group 1:
```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --master_port=60850 assess.py \
  --model_name_or_path_baseline ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --output_response ourbench_group1_responses.json \
  --input_prompts ../../../Datasets/Assess_Gender_Bias/Our_Benchmark_GenderPair/evaluate_prompts/group1/group1_prompts.json \
  --zero_stage 3 \
  --model_type llama2 \
  --max_new_tokens 512 &> ourbench_group1_responses.log  
```

- Group 2:
```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --master_port=60850 assess.py \
  --model_name_or_path_baseline ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --output_response ourbench_group2_responses.json \
  --input_prompts ../../../Datasets/Assess_Gender_Bias/Our_Benchmark_GenderPair/evaluate_prompts/group2/group2_prompts.json \
  --zero_stage 3 \
  --model_type llama2 \
  --max_new_tokens 512 &> ourbench_group2_responses.log
```

- Group 3:  
```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --master_port=60850 assess.py \
  --model_name_or_path_baseline ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --output_response ourbench_group3_responses.json \
  --input_prompts ../../../Datasets/Assess_Gender_Bias/Our_Benchmark_GenderPair/evaluate_prompts/group3/group3_prompts.json \
  --zero_stage 3 \
  --model_type llama2 \
  --max_new_tokens 512 &> ourbench_group3_responses.log
```

#### Step 2: Evaluate bias metrics 

1. Bias-Pair Ratio 
   
For each group, run `evaluate_BPR.py`:

- Group 1: 
```bash
CUDA_VISIBLE_DEVICES=1 deepspeed \
  --master_port=60850 evaluate_BPR.py \
  --model_path ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --model_type llama2_13b \
  --response_file ourbench_group1_responses.json \
  --group_type 1 \
  --zero_stage 3 &> group1_BPR.log
```

- Group 2:
```bash 
CUDA_VISIBLE_DEVICES=1 deepspeed \
  --master_port=60850 evaluate_BPR.py \
  --model_path ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --model_type llama2_13b \
  --response_file ourbench_group2_responses.json \
  --group_type 2 \
  --zero_stage 3 &> group2_BPR.log
```

- Group 3:
```bash
CUDA_VISIBLE_DEVICES=1 deepspeed \
  --master_port=60850 evaluate_BPR.py \
  --model_path ../../../Models/base_models/Meta-Llama-2-13b-chat-hf \
  --model_type llama2_13b \
  --response_file ourbench_group3_responses.json \
  --group_type 3 \
  --zero_stage 3 &> group3_BPR.log  
```

This will generate `group{1,2,3}_BPR_log.txt` with detailed bias-pair ratios per response and `group{1,2,3}_BPR_ratio.json` with the overall ratios.

2. Toxicity 

For each group, run `evaluate_toxicity.py`:

- Group 1:
```bash
python evaluate_toxicity.py ourbench_group1_responses.json ourbench_group1_toxicity.log
```

- Group 2: 
```bash
python evaluate_toxicity.py ourbench_group2_responses.json ourbench_group2_toxicity.log
```

- Group 3:
```bash 
python evaluate_toxicity.py ourbench_group3_responses.json ourbench_group3_toxicity.log
```

This will generate `ourbench_group{1,2,3}_toxicity.log` with average and maximum toxicity scores.

3. Regard

For each group, run `evaluate_regard.py`:

- Group 1:
```bash
python evaluate_regard.py ourbench_group1_responses.json ourbench_group1_regard.log  
```

- Group 2:
```bash
python evaluate_regard.py ourbench_group2_responses.json ourbench_group2_regard.log
```

- Group 3:  
```bash
python evaluate_regard.py ourbench_group3_responses.json ourbench_group3_regard.log
```

This will generate `ourbench_group{1,2,3}_regard.log` with average regard scores (positive, negative, neutral, other).

##  Reduce Gender Bias in LLMs

### Debiasing with Our Proposed Strategy 

Fine-tune the Llama2 model on our debiasing dataset to reduce gender bias. 

1. Navigate to `Code/Reduce_Gender_Bias/Our_Debiasing_Strategy/`
2. Run the debiasing script:
   
```bash
bash run_llama2_13b_debiasing.sh
```

This will generate a debiased version of Llama2-13B in `Models/ft_models/llama2_debiasing/`. 

### GenderPair Results on Debiased Models 

Same steps as assessment, but using the debiased model checkpoints in `Models/ft_models/` to validate bias reduction. Concretely:

1. Generate model responses
   - Use `../../../Models/ft_models/llama2_debiasing` as the `--model_name_or_path_baseline`

2. Evaluate bias metrics 
   - Use `../../../Models/ft_models/llama2_debiasing` as the `--model_path`

### Other Bias Benchmark Results on Debiased Models

Validates bias reduction on the StereoSet, Winoqueer, and BOLD benchmarks. 

#### StereoSet

1. Navigate to `Code/Assess_Gender_Bias/StereoSet/`
2. Run the evaluation script:
   
```bash
CUDA_VISIBLE_DEVICES=1 deepspeed \
  --master_port=60850 evaluate.py \
  --model_path ../../../Models/ft_models/llama2_debiasing \
  --output_file finetuned_resutls.csv \
  --data_path ../../../Datasets/Assess_Gender_Bias/StereoSet/stereoset_gender_intrasentence.csv \
  --zero_stage 3 \
  --model_type llama2 &> finetuned_resutls.log
```

This will generate `finetuned_results.csv` containing the probability distributions over the three options (Stereo_More, Stereo_Less, Unrelated) and the $\Delta$ (difference between Stereo_More and Stereo_Less probability).

#### Winoqueer 

1. Navigate to `Code/Assess_Gender_Bias/Winoqueer/` 
2. Run the evaluation script:
   
```bash
CUDA_VISIBLE_DEVICES=1 deepspeed \
  --master_port=60850 evaluate.py \
  --model_path ../../../Models/ft_models/llama2_debiasing \
  --output_file finetuned_resutls.csv \
  --data_path ../../../Datasets/Assess_Gender_Bias/Winoqueer/winoqueer_gender.csv \
  --zero_stage 3 \
  --model_type llama2 &> finetuned_resutls.log  
```

This will generate `finetuned_results.csv` with the probability distributions over the two options (Stereo_More, Stereo_Less) and the $\Delta$ (difference between Stereo_More and Stereo_Less probability).

#### BOLD

1. Navigate to `Code/Assess_Gender_Bias/BOLD/`
   
2. Generate model responses for each gender group:
   
- Actress group:
```bash 
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --local_rank -1 \
  --master_port=60850 assess.py \
  --model_name_or_path_baseline ../../../Models/ft_models/llama2_debiasing \
  --output_response bold_actresses_responses.json \
  --input_prompts ../../../Datasets/Assess_Gender_Bias/BOLD/bold_gender_actresses_prompts.csv \
  --zero_stage 3 \
  --model_type llama2 \
  --max_new_tokens 128 &> get_bold_actresses_responses.log
```

- Actor group:  
```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed \
  --local_rank -1 \
  --master_port=60850 assess.py \
  --model_name_or_path_baseline ../../../Models/ft_models/llama2_debiasing \
  --output_response bold_actors_responses.json \
  --input_prompts ../../../Datasets/Assess_Gender_Bias/BOLD/bold_gender_actors_prompts.csv \
  --zero_stage 3 \
  --model_type llama2 \
  --max_new_tokens 128 &> get_bold_actors_responses.log
```

3. Evaluate regard scores:
   
```bash
python -u evaluate_regard.py \
  --input_file1 bold_actors_responses.json \
  --input_file2 bold_actresses_responses.json \
  --output_path evaluate_results/ &> evaluate_results/regard.log
```
   
This will generate 4 JSON files under `evaluate_results/`:
- `regard_actor_del.json`: per-instance regard scores for the actor group
- `regard_actress_del.json`: per-instance regard scores for the actress group  
- `regard_actor_avg.json`: average regard scores for the actor group
- `regard_actress_avg.json`: average regard scores for the actress group

## Contact the developers

If you've found a bug or are having trouble getting code to work, please feel free to open an issue on the [<u>GitHub repo</u>](https://github.com/kstanghere/GenderCARE-ccs24). 
