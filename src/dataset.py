
import config
import torch

class ShortJokesDataset:
    def __init__(self, jokes):
        self.jokes = jokes # list of jokes

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, item):
        joke = self.jokes[item]
        joke = "JOKE: " + str(joke) + "<|endoftext|>"
        tokenized_jokes = config.tokenizer.encode_plus(
            joke,
            None,
            add_special_tokens = True,
            max_length = config.MAX_LEN,
            padding = "max_length",
            truncation = True
        )
        ids = tokenized_jokes["input_ids"]
        mask = tokenized_jokes["attention_mask"]
        return {
            "input_ids": torch.tensor(ids, dtype = torch.long),
            "attention_mask": torch.tensor(mask, dtype = torch.long),
            "labels": torch.tensor(ids, dtype = torch.long)
        }
