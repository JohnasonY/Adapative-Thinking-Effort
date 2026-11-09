"""Train with a reproducible holdout split; leave test.csv for evaluation."""

import argparse
from pathlib import Path
import random

from .dataset import load_rows, make_loader
from .scoring import CRITERIA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="models/difficulty_model.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("Epochs and batch size must be positive; learning rate must be in (0, 1).")
    if not 2 <= args.max_length <= 512:
        parser.error("Max length must be between 2 and 512.")
    try:
        rows = load_rows(args.data)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if len(rows) < 2:
        parser.error("At least two rows are needed for the training/validation split.")

    import torch
    from transformers import AutoTokenizer
    from .model import DEFAULT_ENCODER, DifficultyModel

    random.Random(args.seed).shuffle(rows)
    torch.manual_seed(args.seed)
    validation_count = max(1, round(len(rows) * 0.2))
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER)
    train_loader = make_loader(rows[validation_count:], tokenizer, args.batch_size, args.max_length, True)
    validation_loader = make_loader(rows[:validation_count], tokenizer, args.batch_size, args.max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DifficultyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output.stem + "_tokenizer"
    tokenizer.save_pretrained(output.parent / tokenizer_dir)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(**inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for inputs, labels in validation_loader:
                inputs = {key: value.to(device) for key, value in inputs.items()}
                loss = criterion(model(**inputs), labels.to(device))
                validation_loss += loss.item() * len(labels)
        validation_loss /= validation_count
        print(f"Epoch {epoch + 1}: train MSE={train_loss / len(train_loader.dataset):.4f}; "
              f"validation MSE={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "encoder_config": model.encoder.config.to_dict(),
                        "criteria": list(CRITERIA), "max_length": args.max_length,
                        "tokenizer_dir": tokenizer_dir, "seed": args.seed,
                        "validation_mse": best_loss}, output)
    print(f"Best checkpoint saved to {output}")


if __name__ == "__main__":
    main()
