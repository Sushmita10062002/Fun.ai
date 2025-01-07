from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
import pandas as pd
import dataset
import torch
import engine
import config
import os
import pickle

def run(device = "cpu"):
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(config.tokenizer))
    jokes_df = pd.read_csv(config.TRAINING_FILE_PATH)
    jokes_dataset = dataset.ShortJokesDataset(jokes_df["Joke"].tolist())
    train_data_loader = torch.utils.data.DataLoader(
        jokes_dataset, batch_size = config.TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )
    model.to(device)
    num_train_steps = int(len(jokes_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(model.parameters(), lr = 3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = 0, num_training_steps = num_train_steps
    )
    all_losses = {}
    epoch_loss = {}
    for epoch in range(config.EPOCHS):
        print("="*30 + f"EPOCH {epoch + 1}" + "="*30)
        loss, batch_losses = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, epoch)
        model_folder = config.MODEL_FOLDER
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        torch.save(model.state_dict(), os.path.join(model_folder, f"gpt2_joke_generator_{epoch}.pt"))
        all_losses[f"epoch_{epoch}"] = batch_losses
        epoch_loss[f"epoch_{epoch}"] = loss

    with open("../models/all_losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    with open("../models/epoch_losses.pkl", "wb") as f:
        pickle.dump(epoch_loss, f)
if __name__ == "__main__":
    run(device = "cuda")
