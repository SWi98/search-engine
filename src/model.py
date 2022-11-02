import numpy.typing as npt
import numpy as np
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_name: str, device: str = "CPU"):
        self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device)

    def get_embedding(self, X: npt.NDArray[np.string_]):
        return self.model.encode(X)
