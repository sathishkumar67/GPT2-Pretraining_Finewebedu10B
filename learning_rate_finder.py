# imports
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from huggingface_hub import  hf_hub_download
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from schedulefree.adamw_schedulefree import AdamWScheduleFree

file1 = "edufineweb_train_000001.npy"
file2 = "edufineweb_train_000002.npy"
file3 = "edufineweb_train_000003.npy"

# downloading the dataset
for file in [file1, file2, file3]:
    hf_hub_download(repo_id="pt-sk/fineweb_edu_10B", filename=file, repo_type="dataset", local_dir="/kaggle/working/")

# config
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    batch_size: int = 8
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    num_workers: int = 4
    betas = (0.9, 0.95)
    eps = 1e-8
    seed: int = 1337
        
config = GPTConfig()
torch.manual_seed(config.seed)


# dataset preparation
class TokenDataset(Dataset):
    def __init__(self, input_ids, config: GPTConfig):
        self.input_ids = input_ids
        self.block_size = config.block_size

    def __len__(self):
        # Number of full blocks
        return (len(self.input_ids) - 1) // self.block_size

    def __getitem__(self, idx):     
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        x = self.input_ids[start_idx:end_idx]
        y = self.input_ids[start_idx+1:end_idx+1]

        # Pad sequences if they are shorter than block_size
        if len(x) < self.block_size:
            padding_length = self.block_size - len(x)
            x = torch.cat([x, torch.zeros(padding_length, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(padding_length, dtype=torch.long)])
        
        return torch.LongTensor(x.tolist()), torch.LongTensor(y.tolist())
    
tokens1 = np.load(f"/kaggle/working/{file1}")
tokens2 = np.load(f"/kaggle/working/{file2}")
tokens3 = np.load(f"/kaggle/working/{file3}")
tokens = np.concatenate([tokens2, tokens1, tokens3]) # random shuffle

dataset = TokenDataset(tokens, config)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

gpt = GPT(config)

# learning rate values
steps = 18311
lr_values = np.linspace(1e-6, 1e-2, steps)

class GPT2_Wrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.optimizer = self.configure_optimizers()
        self.count = 0

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # accessing the current learning rate and changing it
        current_learning_rate = lr_values[self.count]
        self.count += 1

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

        batch, label = batch
        _, loss = self.model(batch, label)
        self.log("Train_Loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
        return optimizer

gpt_model = GPT2_Wrapper(model=gpt)

# logs
logger = CSVLogger("logs", name="lr_finder", flush_logs_every_n_steps=1)

# setting up the trainer
trainer = Trainer(max_epochs=1,
                  accelerator="cuda",
                  strategy="ddp",
                  devices=2,
                  logger=logger)

# fitting the model
trainer.fit(gpt_model, dataloader)