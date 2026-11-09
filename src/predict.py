"""Load a trained checkpoint and predict five criterion scores."""

from pathlib import Path


class DifficultyPredictor:
    def __init__(self, checkpoint="models/difficulty_model.pt"):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"No trained checkpoint at {path}. Run python -m src.train first.")
        import torch
        from transformers import AutoTokenizer
        from .model import DifficultyModel
        from .scoring import CRITERIA

        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["criteria"] != list(CRITERIA):
            raise ValueError("Checkpoint criteria do not match this project.")
        self.model = DifficultyModel(config=saved["encoder_config"])
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.eval()
        self.max_length = saved["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(path.parent / saved["tokenizer_dir"])

    def predict(self, prompt):
        import torch

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        inputs = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        with torch.inference_mode():
            return self.model(**inputs)[0].tolist()
