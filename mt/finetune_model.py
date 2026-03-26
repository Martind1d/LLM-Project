import torch
import numpy
try:
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except AttributeError:
    pass

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os


from test_finetune import run_sanity_test

MODEL_ID = "tencent/HY-MT1.5-7B"
NEW_LANG = "Swedish"
OUTPUT_DIR = "./tencent_opus_finetune_16r"

LORA_R = 16 # Set quite high as we want to learn a new language
LORA_ALPHA = LORA_R * 2 # good to set 2x the rank
BATCH_SIZE = 16
GRAD_ACCUM = 4 # effective batch size = 64
LEARNING_RATE = 2e-4



def format_instruction(sample):
    """
    Formats the OPUS-100 data into Tencent prompt structure.
    """
    data = sample['translation']
    src = data['en']
    tgt = data['sv']
    text = (
        f"Translate the following segment into {NEW_LANG}, without additional explanation. {src}\n"
        f"{tgt}" 
        f"<|endoftext|>" 
    )

    return text


def bidirectional_exampels(batch):
    
    texts = []

    for data in batch['translation']:
        en_text = data['en']
        sv_text = data['sv']

        prompt_en_sv = (
        f"Translate the following segment into Swedish, without additional explanation. {en_text}\n"
        f"{sv_text}" 
        f"<|endoftext|>" 
    )
        texts.append(prompt_en_sv)

        prompt_sv_en = (
        f"Översätt följande segment till engelska, utan ytterligare förklaring. {sv_text}\n"
        f"{en_text}" 
        f"<|endoftext|>" 
    )
        texts.append(prompt_sv_en)
    return {'text': texts}

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target all linear layers to maximize the learning capacity for a new language 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # load dataset
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-sv")

    #train_dataset = dataset['train'].select(range(50_000))
    print('Preprocessing dataset for bidirectional training')
    train_dataset = dataset['train'].map(
        bidirectional_exampels,
        batched=True,
        remove_columns=['translation'],
        load_from_cache_file=False
    )

    eval_dataset = dataset['validation'].map(
        bidirectional_exampels,
        batched=True,
        remove_columns=['translation'],
        load_from_cache_file=False
    )
    

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-3, # some L2 
        fp16=False,
        bf16=True, # avoid using scaler
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        eval_strategy='steps',
        eval_steps=150,
        logging_steps=50,
        max_length=256, # the opus sentences are short, so 256 is ennough
        packing=True, # dont pack multiple sentences into one block
        dataset_text_field="text", # to find formating
        report_to='none', # maybe want to add weights and biases
        dataloader_num_workers=8,
    )


    # initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        #formatting_func=format_instruction, # use the specific formating fucntion defined above
        args=training_args
    )

    # start training
    print('Start training')
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f'Resuming training from checkpoint: {last_checkpoint}')

    trainer.train(resume_from_checkpoint=last_checkpoint)


    
    run_sanity_test(model, tokenizer, 'tencent')

    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()