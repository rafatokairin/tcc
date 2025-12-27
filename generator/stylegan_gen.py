# =====================================================
# Avaliação de GAN com LPIPS (cross-set) em mamografias
# =====================================================
import csv
import os
import math
import json
import random
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lpips

# -------- CONFIG --------
Z_DIM = 256
W_DIM = 256
N_CLASSES = 2
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_PATH = '../data/ckpt400.pth'
DATA_DIR = '../data'  # Diretório base contendo dataset128 e dataset.csv
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')  # Extensões suportadas
# Quantidade de sintéticas necessárias por classe
TARGET_IMAGES = {
    0: 490,  # Benignas
    1: 490   # Malignas
}

class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.stride = stride
        self.padding = padding
        self.scale = math.sqrt(2) / math.sqrt(in_ch * kernel_size * kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = math.sqrt(2) / math.sqrt(in_dim)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers=4):  # antes 8
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = EqualizedLinear(w_dim, channels)
        self.style_bias = EqualizedLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)

        # Adiciona dimensões extras para broadcasting
        style_scale = self.style_scale(w)[:, :, None, None]
        style_bias = self.style_bias(w)[:, :, None, None]

        return style_scale * x + style_bias

class StyleGAN2GeneratorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, initial_block=False):
        super().__init__()
        self.initial_block = initial_block

        if initial_block:
            self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                EqualizedConv2d(in_ch, out_ch, 3, padding=1),
            )

        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.adain1 = AdaIN(out_ch, w_dim)
        self.adain2 = AdaIN(out_ch, w_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, w):
        batch_size = x.size(0)

        # Noise para primeira camada
        x = self.conv1(x)
        noise1 = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_scale1 * noise1
        x = self.leaky_relu(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        noise2 = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_scale2 * noise2
        x = self.leaky_relu(x)
        x = self.adain2(x, w)

        x = self.leaky_relu(x)
        x = self.adain1(x, w)

        # Noise para segunda camada
        noise2 = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        x = self.conv2(x)
        x = x + self.noise_scale2 * noise2
        x = self.leaky_relu(x)
        x = self.adain2(x, w)

        return x

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, num_classes, out_chan=1, base_ch=64, target_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=8)
        self.initial_constant = nn.Parameter(torch.ones(1, base_ch, 4, 4))

        # Bloco inicial
        self.initial_block = StyleGAN2GeneratorBlock(base_ch, base_ch, w_dim, initial_block=True)

        # Número de etapas até chegar na resolução desejada
        num_layers = int(math.log2(target_size) - 2)  # começa em 4x4

        # Construção automática dos blocos
        channels = base_ch
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            out_ch = max(base_ch // (2 ** (i + 1)), 8)  # não deixar canais muito baixos
            self.blocks.append(StyleGAN2GeneratorBlock(channels, out_ch, w_dim))
            channels = out_ch

        self.to_rgb = EqualizedConv2d(channels, out_chan, 1)

    def forward(self, zc):
        batch_size = zc.size(0)
        w = self.mapping(zc)
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # Bloco inicial
        x = self.initial_block(x, w)
        # Blocos seguintes
        for block in self.blocks:
            x = block(x, w)

        return self.to_rgb(x)

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes).float()

def combine_vectors(x, y):
    return torch.cat((x, y), dim=1)

def find_image_file(base_path, img_id):
    """Tenta encontrar o arquivo de imagem com extensões suportadas"""
    for ext in IMAGE_EXTENSIONS:
        img_path = os.path.join(base_path, f"{img_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def load_dataset_from_csv(csv_path):
    """Carrega o dataset do CSV e retorna lista de (label, image_id)"""
    dataset = []
    label_map = {"BENIGN": 0, "MALIGNANT": 1}
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado em {csv_path}")
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_str = row['pathology'].strip().upper()
            img_id = row['image file path'].strip()
            label = label_map.get(label_str, -1)
            if label != -1:
                dataset.append((label, img_id))
    return dataset

def load_real_images_from_csv(dataset_list, class_label=0, max_images=100):
    """Carrega imagens reais baseado no CSV"""
    real_images = []
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    class_images = [img_id for lbl, img_id in dataset_list if lbl == class_label]
    if not class_images:
        print(f"Nenhuma imagem encontrada para a classe {class_label}")
        return real_images

    chosen_ids = random.sample(class_images, min(max_images, len(class_images)))

    for img_id in tqdm(chosen_ids, desc=f"Carregando reais {class_label}"):
        img_path = find_image_file(os.path.join(DATA_DIR, 'dataset128'), img_id)
        if img_path is None:
            print(f"Imagem não encontrada: {img_id} (tentou extensões: {IMAGE_EXTENSIONS})")
            continue
            
        try:
            img = Image.open(img_path).convert('L')
            img = transform(img).unsqueeze(0)
            real_images.append(img)
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")

    return real_images

def calculate_lpips_cross(real_imgs, fake_imgs, loss_fn, device="cuda"):
    all_min_scores, all_mean_scores = [], []

    if len(real_imgs) == 0:
        print("Nenhuma imagem real encontrada, pulando cálculo de LPIPS.")
        return {
            "min_mean": None,
            "min_std": None,
            "mean_mean": None,
            "mean_std": None,
            "all_min_scores": [],
            "all_mean_scores": []
        }

    for fake in tqdm(fake_imgs, desc="Comparando fakes"):
        fake_rgb = fake.to(device).repeat(1, 3, 1, 1)
        scores = []

        for real in real_imgs:
            real_rgb = real.to(device).repeat(1, 3, 1, 1)
            with torch.no_grad():
                score = loss_fn(real_rgb, fake_rgb).item()
            scores.append(score)

        all_min_scores.append(np.min(scores))
        all_mean_scores.append(np.mean(scores))

    return {
        "min_mean": float(np.mean(all_min_scores)),
        "min_std": float(np.std(all_min_scores)),
        "mean_mean": float(np.mean(all_mean_scores)),
        "mean_std": float(np.std(all_mean_scores)),
        "all_min_scores": all_min_scores,
        "all_mean_scores": all_mean_scores
    }

# =====================================================
# PIPELINE PRINCIPAL
# =====================================================
import os
from torchvision.utils import save_image

LPIPS_THRESHOLD = 0.2  # limite aceitável
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_and_filter_images(gen, dataset_list, loss_fn, class_label, class_name, n_images):
    real_images = load_real_images_from_csv(dataset_list, class_label, 100)
    if not real_images:
        print(f"Nenhuma imagem real disponível para {class_name}.")
        return

    saved_count = 0
    while saved_count < n_images:  # loop infinito até atingir 50 imagens válidas
        # gerar uma imagem
        noise = torch.randn(1, Z_DIM, device=DEVICE)
        label = torch.tensor([class_label], dtype=torch.long, device=DEVICE)
        one_hot = get_one_hot_labels(label, N_CLASSES).to(DEVICE)
        zc = combine_vectors(noise, one_hot)
        with torch.no_grad():
            fake = gen(zc).cpu()

        # calcular LPIPS contra reais
        fake_rgb = fake.to(DEVICE).repeat(1, 3, 1, 1)
        scores = []
        for real in real_images:
            real_rgb = real.to(DEVICE).repeat(1, 3, 1, 1)
            with torch.no_grad():
                score = loss_fn(real_rgb, fake_rgb).item()
            scores.append(score)

        # mean_score = np.mean(scores)
        # vou usar min para ser mais diverso os dados (assemelha a um)
        min_score = np.min(scores)

        if min_score < LPIPS_THRESHOLD:  # salva apenas se estiver abaixo do threshold
            saved_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"{class_name}_{saved_count:03d}.png")
            save_image((fake + 1) / 2, save_path)
            print(f"[{class_name}] Imagem {saved_count} salva (LPIPS={min_score:.3f})")

    print(f"\nTotal de imagens salvas para {class_name}: {saved_count}")


def main():
    # carregar gerador
    gen = Generator(Z_DIM + N_CLASSES, W_DIM, N_CLASSES,
                   out_chan=1, base_ch=32, target_size=IMG_SIZE).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    gen.load_state_dict(ckpt['gen'])
    gen.eval()

    # configurar LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

    # carregar dataset
    dataset_list = load_dataset_from_csv(os.path.join(DATA_DIR, 'dataset.csv'))

    # gerar benignas e malignas
    for class_label, class_name in enumerate(['BENIGN', 'MALIGNANT']):
        n_images = TARGET_IMAGES[class_label]
        generate_and_filter_images(gen, dataset_list, loss_fn, class_label, class_name, n_images)

if __name__ == "__main__":
    main()