import torch
import torch.nn.functional as F
from transformers import AutoTokenizer



def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 200,
             sample: bool = True,
             top_k: int = 40):
    model.eval()

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits, loss = model(indices_to_input)
            next_token_logits = next_token_logits[:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]

        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions
