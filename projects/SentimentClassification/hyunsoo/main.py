from trainer import Trainer
import argparse
from config import Config
import json
import os

def main(args):
    
    if (args.continuous and args.mode == 'train') or args.mode =='test':
        config = Config("/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/model/" + args.name + '.json')
        trainer=Trainer(config, args.mode, args.continuous)
        
    else:
        config = Config("/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/config.json")
        # os.makedirs("/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/model/" + config.model_path[config.model_path.rfind('/')+1:-3])
        name = "/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/model/" + config.model_path[config.model_path.rfind('/')+1:-3] + '.json'

        with open(name, 'w') as f:
            json.dump(config.__dict__, f)

        trainer=Trainer(config, args.mode, args.continuous)
        
    if args.mode == 'train':
        trainer.train()

    else:
        while True:
            text = input('Test sentence>')
            trainer.inference(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--continuous', '-c', action='store_true')
    parser.add_argument('--name', '-n', type=str, default='static_30_filters_300')
    args = parser.parse_args()

    main(args)

