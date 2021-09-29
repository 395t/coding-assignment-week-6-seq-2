from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

import torch
import transformers
from tqdm import tqdm

device = 'cuda'
pretrained_list = ['gpt2', 'gpt2-medium', 'gpt2-large', 'EleutherAI/gpt-neo-1.3B']
pretrained_list = ['EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B']
checkpoint_list = ['save_dir/gpt2-clm/checkpoint-500',
                   'save_dir/gpt2-medium-clm_2e-5_gradacc2_4nodes/checkpoint-500',
                   'save_dir/gpt2-large-clm_1e-5_gradacc1_4nodes/checkpoint-500',
                   'save_dir/gptneo1.3B/checkpoint-500']
checkpoint_list = ['save_dir/gptneo125M/checkpoint-400', 'save_dir/gptneo1.3B/checkpoint-500']

def calculate_ppl(model, tokenizer):
    if hasattr(model.config, 'n_positions'):
        max_length = model.config.n_positions
    else:
        max_length = 1024
    stride = 1024
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print('perplexity is {}'.format(ppl))

for model_id in (pretrained_list + checkpoint_list):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if 'neo' in model_id:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id).to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))
    calculate_ppl(model, tokenizer)
