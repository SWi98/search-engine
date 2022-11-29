import numpy as np
import numpy.typing as npt
import json
from sklearn.neighbors import NearestNeighbors
from src.model import Model



class SearchEngine:
    def __init__(self, embeddings_dir: str, text_dir: str, model: Model):
        self.embeddings = self._read_json(embeddings_dir)
        self.text = self._read_json(text_dir)
        self._fit_knn()
        self.model = model

    def _read_json(self, dir: str) -> npt.NDArray:
        with open(dir, "r", encoding="utf-8") as f:
            return np.array(json.load(f))

    def _fit_knn(self) -> None:
        self.knn = NearestNeighbors(algorithm='brute').fit(self.embeddings)

    def _find_k_closest(self, vector: npt.NDArray, k: int = 1) -> npt.NDArray[np.string_]:
        closest_idxs = self.knn.kneighbors(vector, k)[1][0, :]
        closest_text = self.text[closest_idxs]
        return closest_text

    def run(self, query: str, k_best=1) -> dict[str, str]:
        embedding = self.model.get_embedding(np.array([query]))
        text = self._find_k_closest(vector=embedding, k=k_best).ravel().tolist()
        return text