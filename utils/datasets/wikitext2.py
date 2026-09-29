from datasets import load_dataset

from utils.datasets.registry import register_dataset


@register_dataset("wikitext2")
def get_wikitext2(tokenizer, split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    data = load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split=split)
    return data['text']
