from torch.utils.data import DataLoader

def features_loader_conference(dataset, tokenizer, max_length=128):
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["title"], padding="max_length", truncation=True, max_length=max_length)
    )
    tokenized_dataset.rename_column_("label", "labels")
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return tokenized_dataset


def features_loader_toxicity(dataset, tokenizer, max_length=128):
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["comment_text"], padding="max_length", truncation=True, max_length=max_length)
    )

    def _get_labels(row):
        row["labels"] = [int(row[col]) for col in label_cols]
        return row

    tokenized_dataset = tokenized_dataset.map(lambda x: _get_labels(x))
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return tokenized_dataset


def _collate_fn(tokenizer):
    def _collate(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    return _collate


def get_featurized_dataset(tokenizer, dataset_train, dataset_test, config):
    """Return training and testing dataloaders for already featurized dataset."""
    return DataLoader(dataset_train, collate_fn=_collate_fn(tokenizer), batch_size=config["train_batch_size"],
                      drop_last=True, num_workers=1), \
           DataLoader(dataset_test, collate_fn=_collate_fn(tokenizer), batch_size=config["eval_batch_size"],
                      drop_last=True, num_workers=1)
