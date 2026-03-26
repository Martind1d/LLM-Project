from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import json
import time

import sentencepiece as spm

from datasets import load_dataset

import torch
import math

import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MTModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize model components based on cfg
        self.cfg = cfg

        # Initialize source and target embeddings
        self.src_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.tgt_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Initialize position encodings
        self.src_pos_enc = nn.Embedding(cfg.max_len, cfg.d_model)
        self.tgt_pos_enc = nn.Embedding(cfg.max_len, cfg.d_model)

        # Initialize transformer layers
        self.tr = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.enc_layers,
            num_decoder_layers=cfg.dec_layers,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True)
        
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)
    
    def encode(self, src_ids):
        cfg = self.cfg
        B, S = src_ids.shape
        device = src_ids.device

        src_pad = (src_ids == cfg.pad_id)

        # position ids must be integer
        src_pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S).long()

        src = self.src_emb(src_ids) * math.sqrt(cfg.d_model) + self.src_pos_enc(src_pos_ids)
        memory = self.tr.encoder(src, src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt_in_ids, memory, src_pad):
        cfg = self.cfg
        B, T = tgt_in_ids.shape
        device = tgt_in_ids.device

        tgt_pad = (tgt_in_ids == cfg.pad_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        tgt_pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T).long()

        tgt = self.tgt_emb(tgt_in_ids) * math.sqrt(cfg.d_model) + self.tgt_pos_enc(tgt_pos_ids)

        h = self.tr.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return h


    def forward(self, src, tgt):
        cfg = self.cfg
        B, S = src.shape
        _, T = tgt.shape
        device = src.device

        # Get the padding masks
        src_pad = (src == cfg.pad_id)
        tgt_pad = (tgt == cfg.pad_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # Get the possition indices
        src_pos_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, S)
        tgt_pos_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)

        # Add embeddings and position encodings accroding to Vaswani et al. (2017)
        src_emb = self.src_emb(src) * (cfg.d_model ** 0.5) + self.src_pos_enc(src_pos_ids)
        tgt_emb = self.tgt_emb(tgt) * (cfg.d_model ** 0.5) + self.tgt_pos_enc(tgt_pos_ids)
                                                    
        # Pass through the transformer
        h = self.tr(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
            tgt_mask=tgt_mask
            )
        return self.out(h)

def collate_batch(examples, cfg: DictConfig, sp):
    """Collate a batch of examples for training."""
    pad_id = cfg.pad_id
    src_batch, tgt_batch = [], []
    for ex in examples:
        tr = ex['translation']
        src = [cfg.bos_id] + sp.encode(tr[cfg.src_lang], out_type=int)[: cfg.max_len - 2] + [cfg.eos_id]
        tgt = [cfg.bos_id] + sp.encode(tr[cfg.tgt_lang], out_type=int)[: cfg.max_len - 2] + [cfg.eos_id]
        src_batch.append(torch.tensor(src, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt, dtype=torch.long))
    
    src = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src, tgt

def save_ckpt(path, model, opt, scaler, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(),
         "opt": opt.state_dict(),
         "scaler": scaler.state_dict() if scaler is not None else None,
         "epoch": epoch,
         "step": step},
        path,
    )

@torch.no_grad()
def evaluate_loss(model, loader, criterion, cfg):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        output = model(src, tgt_in)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))

        # token-weighted average (ignoring pad)
        n_tokens = (tgt_out != cfg.pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)

def lr_lambda(current_step, cfg: DictConfig, train_loader):
    """Learning rate scheduler with warmup and linear decay."""
    warmup_steps = cfg.warmup_steps
    
    # linear warmup
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # cosine decay after warmup
    progress = float(current_step - warmup_steps) / float(max(1, cfg.epochs * len(train_loader) - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

@hydra.main(version_base=None,config_path="../conf", config_name="config")
def train_model(cfg: DictConfig):
    # Save the configuration to a file for reference
    os.makedirs(cfg.log_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.log_dir, "config.yaml"))
    log = logging.getLogger(__name__)
    
    log.info("Loading Dataset...")
    ds = load_dataset(cfg.hf_name, cfg.hf_config)
    
    train_ds = ds['train']
    val_ds = ds['validation']

    sp = spm.SentencePieceProcessor(model_file=cfg.sp_model_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: collate_batch(x, cfg, sp)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: collate_batch(x, cfg, sp)
    )

    model = MTModel(cfg).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, cfg, train_loader)
    )

    use_amp = (cfg.amp and DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log.info("Starting training process...")
    log.info("Selected Configurations: \n " + json.dumps(OmegaConf.to_container(cfg), indent=4))

    for epoch in range(1, cfg.epochs + 1):
        step = 0
        total_steps = len(train_loader)
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for src, target in train_loader:
            step +=1
            src = src.to(DEVICE)
            target = target.to(DEVICE)

            #forcedly shift target for teacher forcing
            tgt_input = target[:, :-1]
            tgt_out = target[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()

            if step % cfg.log_interval == 0:
                log.info(f"Epoch [{epoch}/{cfg.epochs}] Learning Rate: {scheduler.get_last_lr()[0]:.6f} Step [{step}/{total_steps}] Loss: {running_loss / cfg.log_interval:.4f}")
                running_loss = 0.0 # Reset running loss for next interval

        log.info(f"Epoch [{epoch}/{cfg.epochs}] completed in {time.time() - start_time:.2f} seconds.")
        log.info("Saving checkpoint...")
        # Checkpoint save
        save_ckpt(
            os.path.join(cfg.save_dir, f"model_epoch{epoch}.pt"),
            model,
            optimizer,
            scaler,
            epoch,
            step,
        )

        # Validation loop
        val_loss = evaluate_loss(model, val_loader, criterion, cfg)
        log.info(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")
    log.info("Training complete. Final model saved.")
    return model



    
if __name__ == "__main__": 
    train_model()