import json
import numpy as np
from tqdm import tqdm
from src.model import Model


class DatasetEmbedder:
    '''
    For a json containing pairs of (question, answer) produces embeddings
    of questions and saves them to a file.
    '''
    def __init__(self, dataset_dir: str, model: Model):
        self.model = model
        self._load_dataset(dataset_dir)

    def _load_dataset(self, dir: str):
        with open(dir, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)

    def embed_dataset(self):
        '''
        Creates embeddings for the questions and saves the dataset
        '''
        self.embeddings = []
        for question in tqdm(self._dataset):
            embedded_question = self.model.get_embedding(np.array([question]))[0].tolist()
            self.embeddings.append(embedded_question)

    def save_data(self, dir: str) -> None:
        with open(dir, "w") as outfile:
            json.dump(self.embeddings, outfile)