import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load
from sklearn.feature_extraction.text import TfidfVectorizer

from huggingface_hub import login

login(token="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")

device = "cuda"

args = parse_args()

if args.dataset_name == "bookcorpus":
  data = torch.load("data/book.pt")["data"]
else:
  data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    tfidf_model = TfidfVectorizer()
    print(f'Number of prompts: {len(data["text"])}')
    tfidf_vector = tfidf_model.fit_transform(data['text'])
    tfidf_array = tfidf_vector.toarray()
    words_set = tfidf_model.get_feature_names_out()
    words_dict = {}
    for i in range(len(words_set)):
        words_dict[words_set[i]] = i
    print(f'Number of words in corpus: {words_set.shape}')
    print(f'TF-IDF matrix shape: {tfidf_array.shape}')
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        middle_size=args.middle_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        importance_filtering = args.enable_global_context
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for i, text in enumerate(data["text"][: args.num_samples]):
    encodings = tokenizer(text, return_tensors="pt")
    # wids: (number of tokens,)
    importance_scores = tfidf_array[i]
    wids = tokenizer(text.split(" "), is_split_into_words = True).word_ids()
    # print(encodings.input_ids.shape)
    # encodings.input_ids[:, :20] = 2048
    if i == 0:
        encodings.input_ids[:, 0] = 2048

    #print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                # set input token importance to -1
                if wids[idx] is None:
                    importance = -1
                else:
                    try:
                        importance = importance_scores[words_dict[text.split(" ")[wids[idx]].lower()]]
                    except:
                        importance = -1
                importance = torch.tensor([importance])
                past_key_values = kv_cache(past_key_values, importance)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")


'''
>>> from transformers import AutoTokenizer
>>> a = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> b = a("I like apples")
>>> b.word_ids()
[None, 0, 0, 0, 0]
>>> b = a("I like apple".split(" "))
>>> b
{'input_ids': [[1, 306], [1, 763], [1, 26163]], 'attention_mask': [[1, 1], [1, 1], [1, 1]]}
>>> b.word_ids()
[None, 0]
>>> b = a("I like apples")
>>> b.word_ids()
[None, 0, 0, 0, 0]
>>> b
{'input_ids': [1, 306, 763, 623, 793], 'attention_mask': [1, 1, 1, 1, 1]}
>>> b = a("I like apples", is_split_into_words = False)
>>> b
{'input_ids': [1, 306, 763, 623, 793], 'attention_mask': [1, 1, 1, 1, 1]}
>>> b.word_ids()
[None, 0, 0, 0, 0]
>>> b = a("I like apple".split(" "), is_split_into_words = True)
>>> b
{'input_ids': [1, 306, 763, 26163], 'attention_mask': [1, 1, 1, 1]}
>>> b.word_ids()
[None, 0, 1, 2]
>>> 
'''