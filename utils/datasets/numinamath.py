from datasets import load_dataset

from utils.datasets.registry import register_dataset
from utils.datasets.common import format_messages


@register_dataset("numinamath")
def get_numinamath(tokenizer, split):
    assert split in ['train', 'test'], f"Unknown split {split} for numinamath"

    def preprocess_fn(example):
        example["messages"] = [
            {
                "content": example["problem"],
                "role": "user",
            },
            {
                "content": example["solution"],
                "role": "assistant",
            }
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = format_messages(example["messages"])
        return {"text": text}

    data = load_dataset("./datasets/NuminaMath-1.5", split=f"train[:256]" if split == "test" else "train[256:]")
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']
