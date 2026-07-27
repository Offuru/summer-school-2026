import os
import math
from pathlib import Path
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from config import Config
import data
import models


def main():

    shakespeare_text = data.load_data()
    shakespeare_encoded, tokenizer = data.encode_text(shakespeare_text)

    model = models.ShakespeareTransformer(
        tokenizer.n_vocab,
        Config.embedding_dim,
        Config.num_heads,
        Config.num_layers,
        Config.block_size,
        Config.dropout,
    ).to(Config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    encoder = tiktoken.get_encoding("gpt2")

    for epoch in tqdm(range(Config.epochs), desc="Training"):

        if epoch % Config.eval_interval == 0:
            model.eval()
            start_idx = encoder.encode("ROMEO: ")
            generated_text = model.generate(
                torch.tensor([start_idx], dtype=torch.long, device=Config.device),
                max_new_tokens=100,
            )[0].tolist()

            model.train()
            print(f"Epoch {epoch}: Generated Text:\n{encoder.decode(generated_text)}\n")
            save_path = Path("saves")
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / f"model_epoch_{epoch}.pt")

        x_batch, y_batch = data.get_batch(shakespeare_encoded)

        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
