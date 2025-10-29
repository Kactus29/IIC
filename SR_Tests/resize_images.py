import os
import torch
import torchvision.transforms as T
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

# === CONFIGURATION ===
input_dir = "../../RAVIR Dataset/train/training_images"
output_dir = "/home/usrs/hnoel/SR_x4/images_upscaled"
os.makedirs(output_dir, exist_ok=True)

# === CHEMIN DU MODÈLE ===
model_path = "/home/usrs/hnoel/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth"

# === CHARGEMENT DU MODÈLE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)

print(f"Chargement du modèle : {model_path}")
loadnet = torch.load(model_path, map_location=device)
model.load_state_dict(loadnet["params_ema"] if "params_ema" in loadnet else loadnet["params"], strict=True)
model.eval().to(device)

# === TRANSFORMATIONS ===
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# === TRAITEMENT DES IMAGES ===
files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
print(f"\n➡️  {len(files)} images trouvées dans {input_dir}\n")

for fname in tqdm(files, desc="Upscaling images"):
    img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
    with torch.no_grad():
        lr = to_tensor(img).unsqueeze(0).to(device)
        sr = model(lr)
        sr_img = to_pil(sr.squeeze().clamp(0, 1).cpu())
    sr_img.save(os.path.join(output_dir, fname))

print(f"\n✅ Super-résolution terminée.\nImages sauvegardées dans : {output_dir}")
