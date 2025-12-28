import os
import json
import torch
import gradio as gr
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# -------------------------------------------------------
# CONFIG — YOUR EXACT PATHS
# -------------------------------------------------------
MODEL_PATH = r"C:\Users\adyab\OneDrive - Mahindra University\Siglip applied ai\SIGLIP.py"
IMAGE_DIR = r"C:\Users\adyab\OneDrive - Mahindra University\applied AI\applied AI\bin-images10k"
META_DIR  = r"C:\Users\adyab\OneDrive - Mahindra University\applied AI\applied AI\metadata10k"
CACHE_PATH = r"C:\Users\adyab\OneDrive - Mahindra University\applied AI\applied AI\embeddings_cache.pt"

DEVICE = "cpu"
torch.set_num_threads(8)

# -------------------------------------------------------
# LOAD SIGLIP MODEL
# -------------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# -------------------------------------------------------
# SIGLIP IMAGE EMBEDDING
# -------------------------------------------------------
def compute_image_embed(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(pixel_values=inputs["pixel_values"])
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

# -------------------------------------------------------
# LOAD / BUILD EMBEDDINGS CACHE
# -------------------------------------------------------
def load_or_build_embeddings():
    dataset_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if os.path.exists(CACHE_PATH):
        try:
            cache = torch.load(CACHE_PATH)
        except:
            cache = {}
    else:
        cache = {}

    missing = [p for p in dataset_paths if p not in cache]

    if missing:
        print(f"[INFO] Embedding {len(missing)} new images...")
        for p in tqdm(missing):
            try:
                cache[p] = compute_image_embed(p)
            except:
                pass
        torch.save(cache, CACHE_PATH)

    return dataset_paths, cache

DATASET_PATHS, DATASET_EMBEDS = load_or_build_embeddings()

# -------------------------------------------------------
# BUILD PRODUCT TABLE (REMOVE DUPLICATES)
# -------------------------------------------------------
PRODUCT_TABLE = []
_seen = set()

for meta_file in os.listdir(META_DIR):
    if not meta_file.lower().endswith(".json"):
        continue

    try:
        meta = json.load(open(os.path.join(META_DIR, meta_file)))
    except:
        continue

    for v in meta.get("BIN_FCSKU_DATA", {}).values():
        name = (v.get("normalizedName") or v.get("name") or "").strip()
        asin = v.get("asin") or v.get("ASIN")
        if name and asin:
            key = (name, asin)
            if key not in _seen:
                PRODUCT_TABLE.append([name, asin])
                _seen.add(key)

GLOBAL_TABLE = PRODUCT_TABLE

# -------------------------------------------------------
# GET QUANTITY FOR PRODUCT
# -------------------------------------------------------
def get_quantity_for_product(image_path, asin):
    base = os.path.splitext(os.path.basename(image_path))[0]
    meta_path = os.path.join(META_DIR, base + ".json")

    if not os.path.exists(meta_path):
        return 0

    try:
        meta = json.load(open(meta_path))
        for v in meta.get("BIN_FCSKU_DATA", {}).values():
            if (v.get("asin") or v.get("ASIN")) == asin:
                return int(v.get("quantity") or v.get("qty") or 1)
    except:
        return 0

    return 0

# -------------------------------------------------------
# MATCHING FUNCTION
# -------------------------------------------------------
def search(product_name, asin, user_qty):

    if not product_name or not asin:
        return ("Select a product row first.", None, {}, "")

    user_qty = int(user_qty)

    candidate_images = []
    for path in DATASET_PATHS:
        base = os.path.splitext(os.path.basename(path))[0]
        meta_path = os.path.join(META_DIR, base + ".json")

        if not os.path.exists(meta_path):
            continue

        meta = json.load(open(meta_path))
        for v in meta.get("BIN_FCSKU_DATA", {}).values():
            if (v.get("asin") or v.get("ASIN")) == asin:
                candidate_images.append(path)
                break

    if not candidate_images:
        return ("No bin contains this product.", None, {}, "")

    query_text = f"{user_qty} {product_name}"
    txt_inputs = processor(text=query_text, return_tensors="pt",
                           truncation=True, padding="max_length",
                           max_length=64).to(DEVICE)

    with torch.no_grad():
        txt_emb = model.get_text_features(input_ids=txt_inputs["input_ids"])
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)

    best_match = None
    best_score = -1e9

    for img in candidate_images:
        score = torch.matmul(txt_emb, DATASET_EMBEDS[img]).item()
        if score > best_score:
            best_score = score
            best_match = img

    qty_in_bin = get_quantity_for_product(best_match, asin)

    warning = ""
    if user_qty > qty_in_bin:
        warning = (
            f"<div style='padding:10px;background:#ffcccc;color:#b30000;"
            f"border:1px solid #b30000;border-radius:6px;font-weight:bold;'>"
            f"⚠ WARNING: Requested quantity ({user_qty}) is greater than available ({qty_in_bin})."
            f"</div>"
        )

    info = (
        f"Matched Image: {os.path.basename(best_match)}\n"
        f"Product: {product_name}\n"
        f"ASIN: {asin}\n"
        f"Quantity in Image: {qty_in_bin}\n"
        f"User Requested: {user_qty}"
    )

    return info, best_match, {
        "product_name": product_name,
        "asin": asin,
        "quantity_in_image": qty_in_bin,
        "user_requested_quantity": user_qty
    }, warning

# -------------------------------------------------------
# ROW SELECT HANDLER (Gradio 6)
# -------------------------------------------------------
def on_select(evt: gr.SelectData):
    idx = evt.index[0]  # row index
    try:
        name, asin = GLOBAL_TABLE[idx]
        return name, asin
    except:
        return "", ""

# -------------------------------------------------------
# UI
# -------------------------------------------------------
with gr.Blocks(title="Smart Bin Classifier") as demo:

    gr.Markdown(
        "<h2 style='text-align:center;'>Smart Bin Classifier</h2>"
    )

    gr.Markdown("### Select the product by clicking on the product name.")

    product_table = gr.Dataframe(
        value=GLOBAL_TABLE,
        headers=["Product Name", "ASIN"],
        wrap=True,
        interactive=True,
        max_height=400,
        show_search="search"
    )

    selected_name = gr.Textbox(label="Selected Product Name", interactive=False)
    selected_asin = gr.Textbox(label="Selected ASIN", interactive=False)

    product_table.select(on_select, outputs=[selected_name, selected_asin])

    qty = gr.Number(label="Requested Quantity", value=1)
    btn = gr.Button("Search Best Bin Image")

    warning_html = gr.HTML()   # warning ABOVE matched info

    out_info = gr.Textbox(label="Matched Info", lines=6)
    out_img = gr.Image(label="Matched Image")
    out_json = gr.JSON(label="Result Details")

    btn.click(
        search,
        inputs=[selected_name, selected_asin, qty],
        outputs=[out_info, out_img, out_json, warning_html]
    )

demo.launch(share=True)
