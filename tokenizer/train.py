import sentencepiece as spm
from datasets import load_dataset
import os
if "corpus.txt" not in os.listdir("./data"):
    ds = load_dataset("Helsinki-NLP/opus-100", "en-sv")
    with open("./data/corpus.txt", "w") as f:
        for item in ds["train"]:
            f.write(item["translation"]["en"] + "\n")
            f.write(item["translation"]["sv"] + "\n")
        spm_input = f.name
else:
    spm_input = "./data/corpus.txt"

print("Training SentencePiece model...")
spm.SentencePieceTrainer.Train(
    input=spm_input,
    model_prefix="./data/tokenizer/spm_en_sv_32k",
    vocab_size=32000,
    character_coverage=0.9995,
    model_type="unigram",
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    )

print("SentencePiece model trained and saved to ./data/tokenizer/")