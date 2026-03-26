import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, StoppingCriteriaList, StoppingCriteria
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from tqdm import tqdm
import hydra 
from omegaconf import OmegaConf
import os
import gc

# Assuming these exist in your project structure
from translate import load_model_for_inference, greedy_translate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Define the Test Lists ---

test_sentences_en_sv = [
    "What time does the train leave tomorrow morning?",
    "I study machine learning and artificial intelligence at the university.", 
    "Can you help me find a good restaurant near the city center?", 
    "The weather was very cold, but the sun was shining all day.",
    "Why is it important to save energy in everyday life?",
    "She forgot her laptop at home and could not work at the office.",
    "What a wholesome ending to this moive although a cheesy one.",
    "The Manager has responsibility for the task, but the CEO holds the accountability for the outcome."
]

test_sentences_sv_en = [
    "Vilken tid landar flyget på Arlanda nästa söndag?",
    "Han forskar om förnybar energi och hållbar utveckling på institutet.",
    "Skulle du kunna rekommendera ett mysigt kafé i närheten av hamnen?",
    "Vinden var otroligt stark, men himlen förblev klar hela eftermiddagen.",
    "Hur påverkar sociala medier ungdomars psykiska hälsa i dagens samhälle?",
    "De tappade nycklarna i snön och lyckades inte låsa upp ytterdörren.",
    "Jag vet inte om jag orkar städa idag, men vi kanske hinner ta en fika.",
    "Min mormor och min farmor diskuterade vems tur det var att bjuda på tretår."
]

PROMPT_TEMPLATES_en_sv = {
    "gpt-sw3": "<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:",
    "custom": None,
    "tencent": "Translate the following segment into Swedish, without additional explanation. {text}\n",
    "helsinki": None 
}

PROMPT_TEMPLATES_sv_en = {
    "gpt-sw3": "<|endoftext|><s>User: Översätt till Engelska från Svenska\n{text}<s>Bot:",
    "custom": None,
    "tencent": "Översätt följande segment till engelska, utan ytterligare förklaring. {text}\n",
    "helsinki": None 
}

class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id


def do_translate_list(model, tokenizer, input_list, translation_type, model_type, cfg, device):
    """
    Iterates over a list of strings and returns a list of translations.
    """
    predictions = []

    if translation_type == 'sv-en':
        prompt_template = PROMPT_TEMPLATES_sv_en[model_type]
    else:
        prompt_template = PROMPT_TEMPLATES_en_sv[model_type]
        
    for src_text in tqdm(input_list, desc=f"Translating {translation_type}"):
        
        pred_text = ""

        # --- Helsinki / OPUS Logic ---
        if model_type == "helsinki":
            if translation_type == 'en-sv':
                model_used = model[0]
                tokenizer_used = tokenizer[0]
            else:
                model_used = model[1]
                tokenizer_used = tokenizer[1]
            
            model_used.eval()
            inputs = tokenizer_used(src_text, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                generated_tokens = model_used.generate(
                    **inputs,
                    max_new_tokens=128,
                    early_stopping=True
                )
            pred_text = tokenizer_used.decode(generated_tokens[0], skip_special_tokens=True)
            pred_text = pred_text.strip()

        # --- Custom Logic ---
        elif model_type == 'custom':
            if translation_type == 'en-sv':
                model_used = model[0]
            else:
                model_used = model[1]
            
            # Assuming greedy_translate handles string input directly
            pred_text = greedy_translate(model_used, tokenizer, cfg, src_text)

        # --- GPT-SW3 Logic ---
        elif model_type == 'gpt-sw3':
            max_model_length = 2048
            prompt = prompt_template.format(text=src_text)
            
            input_tokens = model.tokenizer(src_text, return_tensors='pt').input_ids.to(device)
            dynamic_max_length = max_model_length - input_tokens.shape[1]

            stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=model.tokenizer.bos_token_id)
            
            # Note: Ensure 'model' here is the pipeline or model object as per your original logic
            response = model(
                prompt,
                max_length=dynamic_max_length,
                truncation=True,
                stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
            )
            pred_text = response[0]["generated_text"].split("<s>Bot: ")[-1]

        # --- Tencent / CausalLM Logic ---
        else:
            model.eval()
            prompt = prompt_template.format(text=src_text)
            inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            pred_text = tokenizer.decode(generated_tokens[0][input_len:], skip_special_tokens=True)
            pred_text = pred_text.strip()

        predictions.append(pred_text)

    return predictions


def run_manual_evaluation(model, tokenizer, model_type="custom", cfg=None, device='cuda'):
    print(f"\n----- Manual Translation Check: {model_type} -------")

    # 1. Translate EN -> SV
    print(f"\nrunning EN -> SV...")
    preds_en_sv = do_translate_list(model, tokenizer, test_sentences_en_sv, 'en-sv', model_type, cfg, device)

    # 2. Translate SV -> EN
    print(f"\nrunning SV -> EN...")
    preds_sv_en = do_translate_list(model, tokenizer, test_sentences_sv_en, 'sv-en', model_type, cfg, device)

    # 3. Print Results
    print("\n" + "="*60)
    print(f"RESULTS FOR {model_type.upper()}")
    print("="*60)

    print("\n--- Direction: English -> Swedish ---")
    for src, pred in zip(test_sentences_en_sv, preds_en_sv):
        print(f"Source : {src}")
        print(f"Pred   : {pred}")
        print("-" * 30)

    print("\n--- Direction: Swedish -> English ---")
    for src, pred in zip(test_sentences_sv_en, preds_sv_en):
        print(f"Source : {src}")
        print(f"Pred   : {pred}")
        print("-" * 30)


@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(cfg):
    
    # --- Example: Loading Tencent Model (as per your original code) ---
    print('\nLoading tencent/HY-MT1.5-7B')
    TENCENT_MODEL_ID = "tencent/HY-MT1.5-7B"
    latest_checkpoint = get_last_checkpoint("./tencent_opus_finetune_new")
    
    base_tencent_model = AutoModelForCausalLM.from_pretrained(
        TENCENT_MODEL_ID, 
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    tencent_tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    tencent_model = PeftModel.from_pretrained(base_tencent_model, latest_checkpoint)
    
    # Run the manual check instead of the full dataset eval
    run_manual_evaluation(tencent_model, tencent_tokenizer, model_type='tencent')

    # --- (Optional) Un-comment below to run other models similarly ---
    gc.collect()
    torch.cuda.empty_cache()
    print("\nLoading Helsinki-NLP/opus-mt-en-sv")
    OPUS_en_sv_MODEL_ID = "Helsinki-NLP/opus-mt-en-sv"
    OPUS_sv_en_MODEL_ID = "Helsinki-NLP/opus-mt-sv-en"
    opus_tokenizer_en_sv = AutoTokenizer.from_pretrained(OPUS_en_sv_MODEL_ID)
    opus_tokenizer_sv_en = AutoTokenizer.from_pretrained(OPUS_sv_en_MODEL_ID)
    opus_model_en_sv = AutoModelForSeq2SeqLM.from_pretrained(OPUS_en_sv_MODEL_ID).to(DEVICE)
    opus_model_sv_en = AutoModelForSeq2SeqLM.from_pretrained(OPUS_sv_en_MODEL_ID).to(DEVICE)
    run_manual_evaluation((opus_model_en_sv, opus_model_sv_en), 
                          (opus_tokenizer_en_sv, opus_tokenizer_sv_en), 
                          model_type="helsinki")
    gc.collect()
    torch.cuda.empty_cache()
    gpt_sw3_pipe = pipeline(
    task="text-generation",
    model="AI-Sweden-Models/gpt-sw3-6.7b-v2-translator",
    device=DEVICE)
    run_manual_evaluation(gpt_sw3_pipe, None, 'gpt-sw3')

    gc.collect()
    torch.cuda.empty_cache()
    from translate import load_model_for_inference
    print("\nLoading Baseline Model...")
    baseline_model_en_sv , sp = load_model_for_inference(cfg, f"custom_model_data/en-sv-main.pt")
    baseline_model_sv_en , sp = load_model_for_inference(cfg, f"custom_model_data/sv-en-main.pt")
    run_manual_evaluation((baseline_model_en_sv, baseline_model_sv_en), sp, cfg=cfg)



if __name__ == "__main__":
    main()