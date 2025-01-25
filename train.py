import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from model import SmolLM2
from config import SmolLM2Config
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_length):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.tokens = torch.tensor(self.tokenizer.encode(self.text).ids)
        
    def __len__(self):
        return len(self.tokens) - self.context_length
        
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_length]
        y = self.tokens[idx + 1:idx + self.context_length + 1]
        return x, y

def create_tokenizer(text_path, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    
    # Train tokenizer
    files = [text_path]
    tokenizer.train(files, trainer)
    
    return tokenizer

class SmolLM2Lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = SmolLM2(config)
        self.config = config
        self.step_count = 0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        self.log('train_loss', loss, prog_bar=True)
        
        # Generate text every 500 steps
        self.step_count += 1
        if self.step_count % 500 == 0:
            self.generate_sample()
            
        return loss
    
    def generate_sample(self, max_length=100):
        self.eval()
        with torch.no_grad():
            # Start with a random prompt from the dataset
            context = "First Citizen:"
            input_ids = torch.tensor(self.tokenizer.encode(context).ids).unsqueeze(0).to(self.device)
            
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token = outputs[0, -1].argmax()
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
            generated_text = self.tokenizer.decode(input_ids[0].tolist())
            print(f"\nGenerated text at step {self.step_count}:")
            print(generated_text)
        self.train()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        return optimizer

def main():
    # Setup
    config = SmolLM2Config()
    
    # Create and train tokenizer
    text_path = "input.txt"
    tokenizer = create_tokenizer(text_path, config.vocab_size)
    
    # Create dataset and dataloader
    dataset = TextDataset(text_path, tokenizer, config.context_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = SmolLM2Lightning(config)
    model.tokenizer = tokenizer  # Add tokenizer to model for generation
    
    # Setup logger
    logger = TensorBoardLogger("logs/", name="smollm2")
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='smollm2-{step}',
        save_top_k=1,
        every_n_train_steps=5000,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=5000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        accumulate_grad_batches=4  # Gradient accumulation for larger effective batch size
    )
    
    # Train
    trainer.fit(model, train_loader)
    
    # Save final checkpoint
    trainer.save_checkpoint("checkpoints/final_5000.ckpt")
    
    print("Training completed for 5000 steps. Starting additional 50 steps...")
    
    # Continue training for 50 more steps
    trainer = pl.Trainer(
        max_steps=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    
    # Load checkpoint and continue training
    model = SmolLM2Lightning.load_from_checkpoint("checkpoints/final_5000.ckpt")
    trainer.fit(model, train_loader)
    
    # Save final model after 50 additional steps
    trainer.save_checkpoint("checkpoints/final_5050.ckpt")

if __name__ == "__main__":
    main() 