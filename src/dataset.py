"""CSV loading and lazy batch tokenization."""

import csv
from pathlib import Path

from .scoring import CRITERIA, calculate_difficulty


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"prompt", *CRITERIA}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: prompt,{','.join(CRITERIA)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                prompt = (row["prompt"] or "").strip()
                if not prompt:
                    raise ValueError("Prompt must not be empty.")
                scores = [float(row[key]) for key in CRITERIA]
                calculate_difficulty(scores)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            rows.append((prompt, scores))
    if not rows:
        raise ValueError(f"{path}: dataset is empty.")
    return rows


def make_loader(rows, tokenizer, batch_size=8, max_length=256, shuffle=False):
    import torch
    from torch.utils.data import DataLoader

    def collate(batch):
        prompts, labels = zip(*batch)
        inputs = tokenizer(list(prompts), padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.float32)

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
