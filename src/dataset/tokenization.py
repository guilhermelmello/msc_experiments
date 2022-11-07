from transformers import DataCollatorWithPadding


def get_token_mapper(tokenizer, text_pairs=False):
    def _text_pair_fn(batch):
        return tokenizer(
            text=batch['text'],
            text_pair=batch['text_pair'],
            truncation=True,
            padding=False)
        
    def _text_fn(batch):
        return tokenizer(
            text=batch['text'],
            truncation=True,
            padding=False)

    if text_pairs:
        return _text_pair_fn
    return _text_fn


def tokenize_dataset(
        dataset,
        tokenizer,
        text_pairs=False,
        batch_size=128,
    ):

    text2token = get_token_mapper(
        tokenizer,
        text_pairs=text_pairs
    )
    tokenized_dataset = dataset.map(
        text2token,
        batched=True,
        batch_size=batch_size
    )
    return tokenized_dataset


def to_tf_dataset(
        dataset,
        tokenizer,
        batch_size=128,
        train_split='train',
        dev_split='validation',
    ):

    _kwargs = dict(
        columns=tokenizer.model_input_names,
        label_cols='label',
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors='tf'),
    )

    _kwargs['shuffle'] = True
    train_dataset = dataset[train_split].to_tf_dataset(**_kwargs)

    _kwargs['shuffle'] = False
    dev_dataset = dataset[dev_split].to_tf_dataset(**_kwargs)
    
    return train_dataset, dev_dataset
