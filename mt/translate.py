from train import DEVICE, MTModel
import torch
import sentencepiece as spm

def load_model_for_inference(cfg, ckpt_path: str):
    sp = spm.SentencePieceProcessor(model_file=cfg.sp_model_path)

    model = MTModel(cfg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sp

def qualitative_test(cfg, ckpt_path: str):
    model, sp = load_model_for_inference(cfg, ckpt_path)

    tests = cfg.english_test_questions

    for s in tests:
        sv = greedy_translate(model, sp, cfg, s, max_new_tokens=cfg.max_len)
        print("\nEN:", s)
        print("SV:", sv)


@torch.no_grad()
def greedy_translate(model, sp, cfg, text: str, max_new_tokens: int = 200) -> str:
    model.eval()
    device = next(model.parameters()).device

    # Encode source
    src_ids = [cfg.bos_id] + sp.encode(text, out_type=int)[: cfg.max_len - 2] + [cfg.eos_id]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,S)

    memory, src_pad = model.encode(src)

    # Start target with BOS
    ys = torch.tensor([[cfg.bos_id]], dtype=torch.long, device=device)  # (1,1)

    for _ in range(max_new_tokens):
        h = model.decode(ys, memory, src_pad)               # (1,T,H)
        logits = model.out(h[:, -1, :])                     # (1,V) last position
        next_id = torch.argmax(logits, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)

        if next_id == cfg.eos_id:
            break

    # Decode (optionally drop BOS/EOS)
    out_ids = ys.squeeze(0).tolist()
    # remove BOS
    if out_ids and out_ids[0] == cfg.bos_id:
        out_ids = out_ids[1:]
    # cut at EOS
    if cfg.eos_id in out_ids:
        out_ids = out_ids[: out_ids.index(cfg.eos_id)]

    return sp.decode(out_ids)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import os

    # Load config
    cfg_path = os.path.join("conf", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Path to checkpoint
    ckpt_path = os.path.join(cfg.save_dir, "model_epoch2.pt")

    # Run qualitative test
    qualitative_test(cfg, ckpt_path)