import torch.optim as optim
from cnn import CNN
from tokenizer import Tokenizer
import torch
import torch.nn as nn
import numpy as np
import time
from dataset import NaverDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_


class Trainer:
    def __init__(self, config, mode, continuous):
        self.set_seed()
        self.config = config
        self.freeze_embedding = bool(self.config.freeze_embedding)
        self.tok = Tokenizer(self.config)
        self.pad_id = self.tok.pad_id()
        self.vocab_size = self.tok.get_size()
        self.lr = self.config.lr
        self.clip = self.config.clip
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs

        if bool(self.config.use_pretrained_embedding):
            pretrained_embedding = self.tok.load_pretrained_vectors()

        else:
            pretrained_embedding = None

        self.model = CNN(self.config, self.vocab_size, pad_id=self.pad_id, pretrained_embedding=pretrained_embedding, freeze_embedding=self.freeze_embedding)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if torch.cuda.device_count() > 0:
            print(f"There ars {torch.cuda.device_count()} GPU(s) available.")
            print("Device name:", torch.cuda.get_device_name())
        else:
            print("No GPU avaliable, using the CPU instead.")
            print("Device name:", "cpu")

    
        if mode == 'train':
            # Define optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            # Define criterion
            self.criterion = nn.CrossEntropyLoss()

            if continuous:
        
                check_point = torch.load(self.config.model_path, map_location=self.device)
                self.model.load_state_dict(check_point['model_state_dict'])
                self.optimizer.load_state_dict(check_point['optimzier_state_dict'])

            train_set = NaverDataSet(self.config.train_file, self.tok, self.config.max_len)
            valid_set = NaverDataSet(self.config.valid_file, self.tok, self.config.max_len)
            self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, self.batch_size, shuffle=False)

            print("the mode is train")
            print(f"The number of train data : {len(train_set)}")
            print(f"The number of valid data : {len(valid_set)}")

        else:
            print("the mode is test")
            check_point = torch.load(self.config.model_path, map_location=self.device)
            print(f"Best Epoch : {check_point['epoch']} | Best Loss : {check_point['loss']}")

            self.model.load_state_dict(check_point['model_state_dict'])
            # test_set = NaverDataSet(self.config.test_file, self.tok, self.config.max_len)
            # self.test_loader = DataLoader(test_set, self.batch_size, shuffle=False)

            
            # print(f"The number of train data : {len(test_set)}")



    def set_seed(self, seed_value=42):
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)

    def train(self):
        print("Start training...")
        print(f"The model has {self.model.count_params()}, trainable parameters")
        print("-"*50)

        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(self.epochs):
            self.model.train()
            start_time = time.time()
            total_loss = 0
            for batch in tqdm(self.train_loader, total=len(self.train_loader), desc="Training..."):
                self.optimizer.zero_grad()
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                logits = self.model(inputs)

                loss = self.criterion(logits, labels.view(-1))
                total_loss += loss.item()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
            
            avg_train_loss = total_loss / len(self.train_loader)

            val_loss, val_accuracy = self.valid()

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                print('model saving..')
                torch.save({"model_state_dict":self.model.state_dict(), "optimizer_state_dict":self.optimizer.state_dict, "epoch":epoch+1, "loss":val_loss}, self.config.model_path)

            end_time = time.time()
            time_elapsed = end_time - start_time
            print(f"Epoch {epoch + 1} | Train loss {avg_train_loss:.3f} | Time elapsed : {time_elapsed}")
            print(f"Valid accuracy {val_accuracy:.2f} | Valid loss {val_loss:.3f}")

        print(f"Training complete. \n Best loss : {best_val_loss:.2f} | Best epoch : {best_epoch}")

    def valid(self):

        self.model.eval()
        total_loss = 0
        val_accuracy = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, total=len(self.valid_loader), desc="Validating"):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(inputs)

                loss = self.criterion(logits, labels.view(-1))
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)

                accuracy = (preds == labels.view(-1)).detach().cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

        avg_val_loss = total_loss / len(self.valid_loader)
        val_accuracy = np.mean(val_accuracy)

        return avg_val_loss, val_accuracy

    def inference(self, text):
        input_id = self.tok.encodeAsids(text)
        
        if len(input_id) >= self.config.max_len:
            input_id = input_id[:self.config.max_len]

        else:
            input_id = input_id.tolist() + [self.tok.pad_id()] * (self.config.max_len - len(input_id))

        input_id = torch.LongTensor(input_id).unsqueeze(0)
        logit=self.model(input_id)

        pred = torch.argmax(logit, dim=-1)

        if pred == 0:
            print('negative')

        else:
            print('positive')


        










        