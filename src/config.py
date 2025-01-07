
from transformers import AutoTokenizer

TRAIN_BATCH_SIZE = 32
EPOCHS = 3
MAX_LEN = 64
TRAINING_FILE_PATH = "../inputs/shortjokes.csv"
MODEL_FOLDER = "../models"
MODEL_NAME = "gpt2-medium"
NUM_CORES = 8
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
