import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, StoppingCriteriaList, StoppingCriteria
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
import evaluate
from tqdm import tqdm
import hydra 
from omegaconf import OmegaConf
import os
import pickle

from translate import load_model_for_inference, greedy_translate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

# This class is needed for GPT-SW3 as it has a hard time stopping otherwise
class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id


def do_eval(model, tokenizer, dataset, translation_type, model_type, cfg, device):

    sources = []
    predictions = []
    references = [] 

    if translation_type == 'sv-en':
        src_lang = 'sv'
        ref_lang = 'en'
        prompt_template = PROMPT_TEMPLATES_sv_en[model_type]
    else:
        src_lang = 'en'
        ref_lang = 'sv'
        prompt_template = PROMPT_TEMPLATES_en_sv[model_type]
        
    for data_point in tqdm(dataset, desc=f"Evaluating model"):
            src_text = data_point['translation'][src_lang]
            ref_text = data_point['translation'][ref_lang]

            sources.append(src_text)
            references.append([ref_text]) 

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

            elif model_type=='custom':
                if translation_type == 'en-sv':
                    model_used = model[0]
                else:
                    model_used = model[1]
                
                pred_text = greedy_translate(model_used, tokenizer, cfg, src_text)

            elif model_type =='gpt-sw3':
                max_model_length = 2048
                
                prompt = prompt_template.format(text=src_text)
                
                input_tokens = model.tokenizer(src_text, return_tensors='pt').input_ids.to(device)
                dynamic_max_length = max_model_length - input_tokens.shape[1]

                stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=model.tokenizer.bos_token_id)
                response = model(
                    prompt,
                    max_length=dynamic_max_length,
                    truncation=True,
                    stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
                )
                pred_text = response[0]["generated_text"].split("<s>Bot: ")[-1]

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



    return sources, references, predictions

def run_evaluation(model, tokenizer, dataset, model_type="custom", cfg=None, device='cuda'):
    '''
    Generic eval fucntion, (model_type is custom or hugging face)
    model_type must be: gtp-sw3, custom, tencent or helsinki for the promps templates
    '''
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    comet_metric = evaluate.load("comet")

    print(f"-----Start eval for model type: {model_type}-------")


    test_data = dataset['test'].select(range(10))

    sources_en_sv, references_en_sv, predictions_en_sv = do_eval(model, tokenizer, test_data, 'en-sv', model_type, cfg, device)
    sources_sv_en, references_sv_en, predictions_sv_en = do_eval(model, tokenizer, test_data, 'sv-en', model_type, cfg, device)
    


    bleu_score_sv_en = bleu_metric.compute(predictions=predictions_sv_en, references=references_sv_en)
    bleu_score_en_sv = bleu_metric.compute(predictions=predictions_en_sv, references=references_en_sv)
    avg_bleu = (bleu_score_sv_en['score'] + bleu_score_en_sv['score']) / 2
    # word_order=2 enables chrf++ includes character + n grams
    chrf_score_sv_en = chrf_metric.compute(predictions=predictions_sv_en, references=references_sv_en, word_order=2)
    chrf_score_en_sv = chrf_metric.compute(predictions=predictions_en_sv, references=references_en_sv, word_order=2)
    avg_chrf = (chrf_score_sv_en['score'] + chrf_score_en_sv['score']) / 2
    # the comet wants the references flat 
    comet_score_sv_en = comet_metric.compute(predictions=predictions_sv_en, references=[r[0] for r in references_sv_en], sources=sources_sv_en)
    comet_score_en_sv = comet_metric.compute(predictions=predictions_en_sv, references=[r[0] for r in references_en_sv], sources=sources_en_sv)
    avg_comet = (comet_score_sv_en['mean_score'] + comet_score_en_sv['mean_score']) / 2

    results = {
        "en_sv": {
            "bleu": bleu_score_en_sv['score'],
            "chrf": chrf_score_en_sv['score'],
            "comet": comet_score_en_sv['mean_score']
        },
        "sv_en": {
            "bleu": bleu_score_sv_en['score'],
            "chrf": chrf_score_sv_en['score'],
            "comet": comet_score_sv_en['mean_score']
        },
        "avg": {
            "bleu": avg_bleu,
            "chrf": avg_chrf,
            "comet": avg_comet
        }
    }

    if True:
        print("\n" + "-"*30)
        print(f"Results Summary ({model_type})")
        print("-"*30)
        print(f"Avg BLEU: {results['avg']['bleu']:.2f} | EN->SV: {results['en_sv']['bleu']:.2f} | SV->EN: {results['sv_en']['bleu']:.2f}")
        print(f"Avg COMET: {results['avg']['comet']:.2f} | EN->SV: {results['en_sv']['comet']:.2f} | SV->EN: {results['sv_en']['comet']:.2f}")
        print(f"Avg chrF++: {results['avg']['chrf']:.2f} | EN->SV: {results['en_sv']['chrf']:.2f} | SV->EN: {results['sv_en']['chrf']:.2f}")
        print("-" * 30)

        #Print examples both directions
        print(f"\nExample Translations:")
        
        def print_examples(direction_name, src_list, ref_list, pred_list, num=3):
            print(f"\n--- Direction: {direction_name} ---")
            indexes = np.random.randint(0, len(pred_list), min(num, len(pred_list)))
            for k in indexes:
                print(f"Source: {src_list[k]}")
                print(f"Target: {ref_list[k][0]}") 
                print(f"Pred  : {pred_list[k]}")
                print('-'*20)

        print_examples("EN -> SV", sources_en_sv, references_en_sv, predictions_en_sv, num=2)
        print_examples("SV -> EN", sources_sv_en, references_sv_en, predictions_sv_en, num=2)

    
    return results, (predictions_en_sv, predictions_sv_en)




@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(cfg):

    os.makedirs('eval', exist_ok=True)
    # load dataset 
    dataset = load_dataset("Helsinki-NLP/opus-100", 'en-sv')
    
    
        # the tencent model 
    print('\nLoading tencent/HY-MT1.5-7B')
    TENCENT_MODEL_ID = "tencent/HY-MT1.5-7B"
    latest_checkpoint = get_last_checkpoint("./tencent_opus_finetune_new")
    base_tencent_model = AutoModelForCausalLM.from_pretrained(TENCENT_MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    tencent_tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    tencent_model = PeftModel.from_pretrained(base_tencent_model, latest_checkpoint)
    tencent_model_eval = run_evaluation(tencent_model, tencent_tokenizer, dataset, model_type='tencent')
    with open('eval/tencent_model_eval', 'wb') as tencent_model_filehandler:
        pickle.dump(tencent_model_eval, tencent_model_filehandler)



    print("\nLoading Baseline Model...")
    baseline_model_en_sv , sp = load_model_for_inference(cfg, f"custom_model_data/en-sv-main.pt")
    baseline_model_sv_en , sp = load_model_for_inference(cfg, f"custom_model_data/sv-en-main.pt")
    custom_model_eval = run_evaluation((baseline_model_en_sv, baseline_model_sv_en) , sp, dataset, model_type='custom', cfg=cfg)
    with open('eval/custom_model_eval', 'wb') as custom_model_filehandler:
        pickle.dump(custom_model_eval, custom_model_filehandler)
    
    

    print("\nLoading Helsinki-NLP/opus-mt-en-sv")
    OPUS_en_sv_MODEL_ID = "Helsinki-NLP/opus-mt-en-sv"
    OPUS_sv_en_MODEL_ID = "Helsinki-NLP/opus-mt-sv-en"
    opus_tokenizer_en_sv = AutoTokenizer.from_pretrained(OPUS_en_sv_MODEL_ID)
    opus_tokenizer_sv_en = AutoTokenizer.from_pretrained(OPUS_sv_en_MODEL_ID)
    opus_model_en_sv = AutoModelForSeq2SeqLM.from_pretrained(OPUS_en_sv_MODEL_ID).to(DEVICE)
    opus_model_sv_en = AutoModelForSeq2SeqLM.from_pretrained(OPUS_sv_en_MODEL_ID).to(DEVICE)
    opus_model_eval = run_evaluation((opus_model_en_sv, opus_model_sv_en),
                                     (opus_tokenizer_en_sv, opus_tokenizer_sv_en),
                                       dataset, model_type="helsinki")
    with open('eval/opus_model_eval', 'wb') as opus_model_filehandler:
        pickle.dump(opus_model_eval, opus_model_filehandler)

    
    # add the ai sweden model
    print("\nLoading AI-Sweden-Models/gpt-sw3-6.7b-v2-translator")
    gpt_sw3_pipe = pipeline(
    task="text-generation",
    model="AI-Sweden-Models/gpt-sw3-6.7b-v2-translator",
    device=DEVICE)
    swe_model_eval = run_evaluation(gpt_sw3_pipe, None, dataset, model_type="gpt-sw3")
    with open('eval/swe_model_eval', 'wb') as swe_model_eval_model_filehandler:
        pickle.dump(swe_model_eval, swe_model_eval_model_filehandler)
        
    

    print("Running without the finetuning:")
    with tencent_model.disable_adapter():
        tencent_model_eval = run_evaluation(tencent_model, tencent_tokenizer, dataset, model_type='tencent')
    with open('eval/original_tencent_model_eval', 'wb') as tencent_model_filehandler:
        pickle.dump(tencent_model_eval, tencent_model_filehandler)


if __name__ == "__main__":
    main()