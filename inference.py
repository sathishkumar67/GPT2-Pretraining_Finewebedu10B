from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt2 import gpt2
from generate import *

from huggingface_hub import login, hf_hub_download

def main():
    # getting the login tokens from the user
    login_token = input("Enter your hugging face login token: ")

    # logging in to the hugging face account
    login(login_token)

    # downloading the model from the hugging face hub
    hf_hub_download(repo_id="pt-sk/GPT2_pretrained_finewebedu10B", filename="finewebedu_gpt.pt", repo_type="model", local_dir="/kaggle/working/")

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # loading the model
    gpt2.load_state_dict(torch.load("/kaggle/working/finewebedu_gpt.pt"))
    
    return gpt2, tokenizer, generate


if __name__ == "__main__":
    main()