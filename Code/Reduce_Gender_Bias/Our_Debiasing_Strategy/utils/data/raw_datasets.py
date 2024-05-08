from datasets import load_dataset
from torch.utils.data import Subset
import re
import json


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class Llama2Debiasing(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "llama2/debiasing"
        self.dataset_name_clean = "llama2_debiasing"
        
        # Load train and test data from the json files
        with open("../../../Datasets/Reduce_Gender_Bias/Our_Debiasing_Dataset/merged_3groups_ft.json", "r") as f:
            train_data = json.load(f)
            
        with open("../../../Datasets/Reduce_Gender_Bias/Our_Debiasing_Dataset/merged_3groups_ft_validate.json", "r") as f:
            test_data = json.load(f)
        
        # Create dictionary containing the train and test data
        self.raw_datasets = {"train": train_data, "test": test_data}

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return "<s>[INST] " + sample['prompt'] + " [/INST] "

    def get_chosen(self, sample):
        return sample['sentence']+ " </s>"

    def get_rejected(self, sample):
        return None

    def get_prompt_and_chosen(self, sample):
        return "<s>[INST] " + sample['prompt'] + " [/INST] "+ sample['sentence']+ " </s>"

    def get_prompt_and_rejected(self, sample):
        return None

