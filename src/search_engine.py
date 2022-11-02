import numpy as np
import numpy.typing as npt
import json
from sklearn.neighbors import NearestNeighbors
from src.model import Model



class SearchEngine:
    def __init__(self, embeddings_dir: str, qa_dir: str, model: Model):
        self.embeddings = self._read_json(embeddings_dir)
        self.qa = self._read_json(qa_dir)
        self._fit_knn()
        self.model = model

    def _read_json(self, dir: str) -> npt.NDArray:
        with open(dir, "r", encoding="utf-8") as f:
            return np.array(json.load(f))

    def _fit_knn(self) -> None:
        self.knn = NearestNeighbors(algorithm='brute').fit(self.embeddings)

    def _find_k_closest(self, vector: npt.NDArray, k: int = 1) -> npt.NDArray[np.string_]:
        closest_idxs = self.knn.kneighbors(vector, k)[1][0, :]
        closest_qa = self.qa[closest_idxs]
        return closest_qa

    def run(self, query: str) -> dict[str, str]:
        embedding = self.model.get_embedding(np.array([query]))
        closest_qa = self._find_k_closest(vector=embedding, k=1).ravel().tolist()
        return {"question": closest_qa[0], "answer": closest_qa[1]}