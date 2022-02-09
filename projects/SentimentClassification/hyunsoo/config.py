import json

class Config:
    def __init__(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            self.__dict__.update(data)

if __name__ == "__main__":
    config = Config('./projects/SentimentClassification/hyunsoo/config.json')
    print(config.vocab_path)
        