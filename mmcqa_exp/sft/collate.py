"""Chat-template batching with the loss on the answer tokens only.

Text is built as ``<inference prompt> + answer + <|im_end|>``, where the inference prompt is exactly
what ``apply_chat_template(add_generation_prompt=True, enable_thinking=False)`` produces, so training
sees the same prefix as inference (including the empty think block). The answer start is found by
locating the template's assistant header inside the token ids.
"""

import torch
from PIL import Image

END = "<|im_end|>"


def find_last(seq: list, sub: list) -> int:
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if seq[i:i + n] == sub:
            return i
    return -1


def prompt_text_for(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


class SFTCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "right"
        dummy = prompt_text_for(processor, "x")
        header = dummy[dummy.rindex("<|im_start|>assistant"):]
        self.header_ids = self.tokenizer(header, add_special_tokens=False)["input_ids"]

    def build_texts(self, example: dict) -> str:
        return prompt_text_for(self.processor, example["prompt"]) + example["answer"].strip() + END

    def __call__(self, batch: list) -> dict:
        images = []
        for ex in batch:
            with Image.open(ex["image"]) as source:
                images.append(source.convert("RGB"))
        texts = [self.build_texts(ex) for ex in batch]
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        input_ids, mask = enc["input_ids"], enc["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        for i in range(input_ids.shape[0]):
            ids = input_ids[i].tolist()
            at = find_last(ids, self.header_ids)
            if at < 0:
                raise RuntimeError("Assistant header not found in the tokenized example; the chat template changed.")
            start, end = at + len(self.header_ids), int(mask[i].sum())
            if end <= start:
                raise RuntimeError("Example has no answer tokens (truncated?).")
            labels[i, start:end] = input_ids[i, start:end]
        enc["labels"] = labels
        return dict(enc)
