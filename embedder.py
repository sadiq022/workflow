from transformers import AutoTokenizer, AutoModel
import torch


class Embedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts):
        embeddings = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                output = self.model(**inputs).last_hidden_state
                pooled = output.mean(dim=1)

            embeddings.append(pooled[0].cpu().numpy().tolist())

        return embeddings
