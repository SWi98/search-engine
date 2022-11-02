import json
from tqdm import tqdm
from typing import Union


class DataPreprocess:
    def __init__(self, source_dir: str):
        self._load_data(source_dir)

    def _load_data(self, dir: str) -> None:
        with open(dir, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def preprocess_data(self, limit: Union[None, int]) -> None:
        '''
        Preprocess the data to a form of a list of pairs (lists): [question, answer]
        '''
        self.dataset = []
        for i, line in tqdm(enumerate(self.lines)):
            if limit is not None and i >= limit:
                break
            question, answer = line.split("===>")
            answer = answer.strip()
            self.dataset.append([question, answer])

    def save_data(self, dir: str) -> None:
        with open(dir, "w", encoding="utf-8") as outfile:
            json.dump(self.dataset, outfile, ensure_ascii=False)
