"""SapBERT encoder with mean-token pooling for biomedical entity linking."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class SapBERTEncoder:
    """Encode text using SapBERT with mean-token pooling and L2 normalization."""

    def __init__(self, model_name: str):
        print(f"Loading SapBERT model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).cuda().eval()
        print("  SapBERT loaded on GPU.")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts using mean-token pooling, L2-normalized."""
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=(len(texts) + batch_size - 1) // batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=128,
            ).to("cuda")

            with torch.no_grad():
                output = self.model(**encoded)
                # Mean pooling over non-padding tokens
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                token_embeds = output.last_hidden_state
                mean_embeds = (token_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                # L2 normalize so IP == cosine similarity
                mean_embeds = torch.nn.functional.normalize(mean_embeds, dim=1)
                all_embeddings.append(mean_embeds.cpu().float().numpy())

        return np.ascontiguousarray(np.vstack(all_embeddings), dtype=np.float32)

    def close(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
