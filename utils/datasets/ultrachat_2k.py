from datasets import load_dataset

from utils.datasets.registry import register_dataset
from utils.datasets.common import format_messages


@register_dataset("ultrachat_2k")
def get_ultrachat_2k(tokenizer, split):
    assert split in ['train', 'test'], f"Unknown split {split} for ultrachat_2k"

    def preprocess_fn(example):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = format_messages(example["messages"])
        return {"text": text}

    data = load_dataset("./datasets/ultrachat_2k", split=f"train_sft[:128]" if split == "test" else "train_sft[128:]")
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']
