import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL_ID = "tencent/HY-MT1.5-7B"
ADAPTER_PATH = "./tencent_opus_finetune_new/checkpoint-4500" # Path where your trainer saved the model
NEW_LANG = "Swedish"

def load_model():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    
    # 1. Load Base Model in 4-bit (Same as training)
    # This is necessary to fit the 7B model + adapters on your GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Optimized for your L4 GPU
        bnb_4bit_use_double_quant=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load & Attach LoRA Adapters
    print(f"Loading Adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() 
    
    return model, tokenizer

def run_sanity_test(model, tokenizer, model_type):
    '''
    Run a sanity test on some examples
    '''
    print("\n" + "="*50)
    print("Running sanity test on model ", model_type)
    print("="*50)

    test_senctences = [
        "Hello",
        "The cat sat on the mat with his hat and looked at the sun.",
        "Artificial intelligence is a cool new technology which is fun to work with.",
        "This course has taught me a ton of new things",
        "Evaluation of different translation models is what we have chosen to do for this project.",
        "What a wholesome ending to this moive although a cheesy one.",
        "Well that was an awkward interaction.",
        "The Manager has responsibility for the task, but the CEO holds the accountability for the outcome."
    ]

    test_senctences_en_sv = [
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
    model.eval()

    for trans in ['sv_en', 'en_sv']:
        for text in test_senctences:


            if model_type == "helsinki":
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to('cuda')
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        early_stopping=True
                    )
                generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                

            else:
                prompt = f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:"

                inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False, # I want the greedy answer, as we are not looking for a creative model
                        pad_token_id=tokenizer.eos_token_id
                                    )
                
                # slice to remove the prompt from the output
                input_len = inputs.input_ids.shape[1]
                input_len = 0
                generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                generated_text = generated_text.strip()


            print(f"English: {text}")
            print(f"Swedish: {generated_text}")
            print("-" * 25)

def main():
   

    model, tokenizer = load_model()
    run_sanity_test(model, tokenizer, 'Finetuned')


    OPUS_MODEL_ID = "Helsinki-NLP/opus-mt-en-sv"
    opus_tokenizer = AutoTokenizer.from_pretrained(OPUS_MODEL_ID)
    opus_model = AutoModelForSeq2SeqLM.from_pretrained(OPUS_MODEL_ID).to('cuda')
    run_sanity_test(opus_model, opus_tokenizer, 'helsinki')

    GPT_SW3_ID = "AI-Sweden-Models/gpt-sw3-6.7b-v2-translator"
    sw3_tokenizer = AutoTokenizer.from_pretrained(GPT_SW3_ID)
    sw3_model = AutoModelForCausalLM.from_pretrained(GPT_SW3_ID).to('cuda')
    run_sanity_test(sw3_model, sw3_tokenizer, 'sw3_model')

    from translate import load_model_for_inference
    print("\nLoading Baseline Model...")
    baseline_model_en_sv , sp = load_model_for_inference(cfg, f"custom_model_data/en-sv-main.pt")
    baseline_model_sv_en , sp = load_model_for_inference(cfg, f"custom_model_data/sv-en-main.pt")
    custom_model_eval = run_evaluation((baseline_model_en_sv, baseline_model_sv_en) , sp, dataset, model_type='custom', cfg=cfg)
'''
    print("Running without the finetuning:")
    with model.disable_adapter():
        run_sanity_test(model, tokenizer)
'''

if __name__ == "__main__":
    main()