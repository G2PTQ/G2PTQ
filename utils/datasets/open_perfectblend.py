"""Loader for the mlabonne/open-perfectblend instruction dataset."""

from datasets import load_dataset

from utils.datasets.common import format_messages, sample_and_tokenize
from utils.datasets.registry import register_dataset


# Open-PerfectBlend stores conversations in ShareGPT format.  Keep the
# aliases here so rows from the component datasets that already use the
# canonical ``user``/``assistant`` names are handled as well.
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _to_messages(conversations):
    """Convert a ShareGPT conversation to tokenizer chat-template messages."""

    messages = []
    for turn in conversations:
        sender = turn["from"].strip().lower()
        try:
            role = _ROLE_MAP[sender]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Open-PerfectBlend conversation role {turn['from']!r}; "
                f"expected one of {sorted(_ROLE_MAP)}"
            ) from exc
        messages.append({"role": role, "content": turn["value"]})
    return messages


@register_dataset("open_perfectblend", sampler=sample_and_tokenize)
@register_dataset("open-perfectblend", sampler=sample_and_tokenize)
def get_open_perfectblend(tokenizer, split):
    """Return formatted Open-PerfectBlend texts for ``split``.
    """
    assert split in ["train"], f"Unknown split {split} for open_perfectblend"

    def preprocess_fn(example):
        messages = _to_messages(example["conversations"])
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = format_messages(messages)
        return {"text": text}

    data = load_dataset(
        "./datasets/open-perfectblend",
        split="train",
    )
    # data = data.shuffle(seed=42).select(range(10000))
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data["text"]
