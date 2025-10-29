import os
import cv2
from tqdm import tqdm

# === CONFIGURATION ===
input_dir = "../../RAVIR Dataset/train/training_masks"
output_dir = "/home/usrs/hnoel/SR_x4/masks_resized"
scale = 4
os.makedirs(output_dir, exist_ok=True)

# === TRAITEMENT DES MASQUES ===
files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
print(f"\n➡️  {len(files)} masques trouvés dans {input_dir}\n")

for fname in tqdm(files, desc="Resizing masks"):
    img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    resized = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_dir, fname), resized)

print(f"\n✅ Redimensionnement terminé.\nMasques sauvegardés dans : {output_dir}")
