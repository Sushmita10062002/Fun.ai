import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
model.resize_token_embeddings(len(config.tokenizer))

model_path = "../models/gpt2_joke_generator_0.pt"
model.load_state_dict(torch.load(model_path))

device = "cuda"
model.to(device)

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def predict(length_of_joke, number_of_jokes):
  joke_num = 0
  model.eval()

  with torch.no_grad():
      for joke_idx in range(number_of_jokes):
          joke_finished = False
          cur_ids = torch.tensor(config.tokenizer.encode('JOKE: How do you feel when you lie to me?')).unsqueeze(0).to(device)
          for i in range(length_of_joke):
              outputs = model(cur_ids, labels = cur_ids)
              loss, logits = outputs[:2]
              softmax_logits = torch.softmax(logits[0, -1], dim = 0)
              if i < 3:
                n = 20
              else:
                n = 3
              next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n = n)
              cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim = 1)

              if next_token_id in config.tokenizer.encode("<|endoftext|>"):
                joke_finished = True
                break

          if joke_finished:
              joke_num = joke_num + 1
              output_list = list(cur_ids.squeeze().to("cpu").numpy())
              output_text = config.tokenizer.decode(output_list)
              print(output_text + "\n")

predict(30, 2)