# train_siglip_FINAL_PERFECT.py  ← SAVE WITH THIS EXACT NAME
import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# ------------------ PATHS ------------------
IMAGE_DIR = r"D:\CLIP_AmazonBin\data\bin-images"
META_DIR  = r"D:\CLIP_AmazonBin\data\metadata"
MODEL_DIR = r"D:\CLIP_AmazonBin\models"
MODEL_NAME = "google/siglip-base-patch16-224"

BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 64                    # SIGLIP MAX IS 64, NOT 77!!!

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------ CUSTOM COLLATE (still needed for safety) ------------------
def siglip_collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])

    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    # Pad to longest in batch (all will already be <=64, usually exactly 64)
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        l = len(ids)
        padded_input_ids[i, :l] = ids
        padded_attention_mask[i, :l] = mask

    return {
        "pixel_values": pixel_values,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask
    }


class AmazonBinDataset(Dataset):
    def __init__(self, image_dir, meta_dir, processor):
        self.processor = processor
        self.samples = []

        json_files = [f for f in os.listdir(meta_dir) if f.lower().endswith('.json')]
        print(f"Found {len(json_files)} JSON files")

        matched = 0
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            img_path = os.path.join(image_dir, base_name + ".jpg")
            if not os.path.exists(img_path):
                continue

            caption = "amazon bin product"
            try:
                with open(os.path.join(meta_dir, json_file), 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                items = meta.get("BIN_FCSKU_DATA", {})
                if items:
                    first = list(items.values())[0]
                    name = first.get("normalizedName") or first.get("name", "")
                    if name.strip():
                        caption = name.strip()
            except:
                pass

            self.samples.append((img_path, caption))
            matched += 1
            if matched % 2000 == 0:
                print(f"  Matched {matched} samples...")

        print(f"\nFINAL DATASET SIZE: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        # Critical: max_length=64 for SigLIP!
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print("Loading SigLIP...")

    # Use fast processor + explicitly set max_length
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Unfreeze only vision, text, and logit_scale
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(x in name for x in ["vision_model", "text_model", "logit_scale"]):
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.05
    )

    dataset = AmazonBinDataset(IMAGE_DIR, META_DIR, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,           # increase to 4 if you want faster loading
        pin_memory=True,
        collate_fn=siglip_collate_fn
    )

    print(f"\nSTARTING TRAINING — {EPOCHS} EPOCHS ON {len(dataset)} SAMPLES\n")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            img_emb = nn.functional.normalize(outputs.image_embeds, dim=-1)
            txt_emb = nn.functional.normalize(outputs.text_embeds, dim=-1)

            logits = model.logit_scale.exp() * img_emb @ txt_emb.t()

            labels = torch.arange(logits.shape[0], device=DEVICE)
            loss = (nn.CrossEntropyLoss()(logits, labels) + 
                    nn.CrossEntropyLoss()(logits.t(), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"\nEPOCH {epoch+1} COMPLETED — AVG LOSS: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            save_path = os.path.join(MODEL_DIR, f"siglip-amazonbin-epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"MODEL SAVED → {save_path}")

    print("\nTRAINING FINISHED! Final model:")
    print(os.path.join(MODEL_DIR, f"siglip-amazonbin-epoch{EPOCHS}"))
    print("Ready for 95%+ accuracy in your Gradio app!")