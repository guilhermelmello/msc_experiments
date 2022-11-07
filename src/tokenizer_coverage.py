from transformers import AutoTokenizer

import numpy as np


def get_tokenizer_fn(tokenizer, text_pairs=False):
    def text_pair_fn(batch):
        return tokenizer(
            text=batch['text'],
            text_pair=batch['text_pair'],
            padding=True)
        
    def text_fn(batch):
        return tokenizer(
            text=batch['text'],
            padding=True)

    if text_pairs:
        return text_pair_fn
    return text_fn


def get_unk_count_fn(tokenizer):
    def unk_count_fn(batch):
        input_ids = np.array(batch['input_ids'])
        is_unk_token = input_ids == tokenizer.unk_token_id
        unk_count = np.sum(is_unk_token, axis=-1)
        return dict(unk_tokens=unk_count)

    return unk_count_fn
    

def compute_dataset_coverage(data, model_ids, text_pairs=False, splits=['train', 'validation']):
    if isinstance(model_ids, str):
        model_ids = [model_ids]

    results = dict()
    for model_id in model_ids:
        print("USING MODEL:", model_id)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # dataset tokenization
        _to_tokens = get_tokenizer_fn(tokenizer, text_pairs=text_pairs)
        tokenized_ds = data.map(_to_tokens, batched=True)

        # creating column with unk count
        _unk_count = get_unk_count_fn(tokenizer)
        tokenized_ds = tokenized_ds.map(_unk_count, batched=True)

        unk_tokens = 0
        for split in splits:
            unk_tokens += np.sum(tokenized_ds[split]['unk_tokens'])

        print("Model:", model_id)
        print("Encontrou:", unk_tokens)

        results[model_id] = unk_tokens

    return results