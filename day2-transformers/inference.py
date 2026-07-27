import torch
import tiktoken
from config import Config
import models

encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab

model = models.ShakespeareTransformer(
    vocab_size,
    Config.embedding_dim,
    Config.num_heads,
    Config.num_layers,
    Config.block_size,
    Config.dropout,
).to(Config.device)

checkpoint = torch.load("saves/model_epoch_100.pt", map_location=Config.device)
model.load_state_dict(checkpoint)

model.eval()

prompt = "<insert your initial prompt here>"
starts_ids = encoder.encode(prompt)

context = torch.tensor(starts_ids, dtype=torch.long, device=Config.device).unsqueeze(0)

with torch.no_grad():
    generated_ids = model.generate(context, max_new_tokens=100)[0].tolist()

print(encoder.decode(generated_ids))
