from datasets import load_dataset

from utils.datasets.registry import register_dataset


@register_dataset("neuralmagic")
def get_neuralmagic(tokenizer, split):
    assert split in ['train'], "NeuralMagic only has a train split"

    def preprocess_fn(example):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            text = example["text"]
        return {"text": text}

    data = load_dataset("./datasets/LLM_compression_calibration", split=split)
    data = data.map(preprocess_fn, remove_columns=data.column_names)
    return data['text']
