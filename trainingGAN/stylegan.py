from torch.cuda.amp import autocast, GradScaler
from google.colab import drive
drive.mount('/content/drive')
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

# -----------------------
# Configurações principais
# -----------------------
CSV_PATH   = r'/content/drive/MyDrive/training/dataset.csv'
ROOT_DIR   = r'/content/drive/MyDrive/training/dataset'
IMG_SIZE   = 128              # começa em 128x128 p/ acelerar
Z_DIM      = 256              # reduzido (antes 512)
W_DIM      = 256              # reduzido (antes 512)
N_CLASSES  = 2
BATCH_SIZE = 8                # reduzido p/ VRAM do Colab
N_EPOCHS   = 400
LR_G       = 0.0025
LR_D       = 0.0025
BETAS      = (0.0, 0.99)
R1_GAMMA   = 10.0
ADA_TARGET = 0.6
ADA_INTERVAL = 4
ADA_SPEED  = 0.01
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = (DEVICE == 'cuda')

os.makedirs('/content/drive/MyDrive/training/progress', exist_ok=True)
os.makedirs('/content/drive/MyDrive/training/checkpoints', exist_ok=True)

# -----------------------
# Dataset
# -----------------------
class CustomMedicalDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None,
                 benign_token='BENIGN', path_col='image file path', pathology_col='pathology', ext='.jpg'):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.data['label'] = self.data[pathology_col].astype(str).apply(
            lambda x: 0 if benign_token in x.upper() else 1
        )
        self.path_col = path_col
        self.ext = ext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = str(row[self.path_col]) + self.ext
        img_path = os.path.join(self.root_dir, filename)

        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

ada_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5, fill=0),
])

dataset = CustomMedicalDataset(CSV_PATH, ROOT_DIR, transform=transform)
print(f"Tamanho do dataset: {len(dataset)} imagens")

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=PIN_MEMORY, drop_last=True
)

# -----------------------
# Utilidades
# -----------------------
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes).float()

def combine_vectors(x, y):
    return torch.cat((x, y), dim=1)

def get_input_dimensions(z_dim, n_classes, im_chan=1):
    gen_in = z_dim + n_classes
    disc_in = im_chan + n_classes
    return gen_in, disc_in

# -----------------------
# Blocos de rede (reduzidos)
# -----------------------
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

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(2),
            EqualizedConv2d(in_ch, out_ch, 1)
        )
        self.block = nn.Sequential(
            EqualizedConv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )
        self.alpha = 0.0  # Para progressive growing

    def forward(self, x):
        if self.alpha < 1.0:
            return self.alpha * self.block(x) + (1 - self.alpha) * self.residual(x)
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, in_chan, n_classes, base_ch=64, target_size=256):
        super().__init__()
        n_blocks = int(math.log2(target_size)) - 2
        ch = base_ch

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_chan + n_classes, ch, 1),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(DiscriminatorBlock(ch, min(512, ch * 2)))
            ch = min(512, ch * 2)

        self.final_block = nn.Sequential(
            EqualizedConv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(ch, 1, 4, padding=0)
        )

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return self.final_block(x).view(-1)

# -----------------------
# Adaptive Augmentation (ADA)
# -----------------------
class AdaptiveAugment:
    def __init__(self, ada_target=0.6, ada_speed=0.01, batch_size=BATCH_SIZE):
        self.ada_target = ada_target
        self.ada_speed = ada_speed
        self.batch_size = batch_size
        self.ada_p = 0.0
        self.ada_r_t = torch.tensor(0.0, device=DEVICE)
        self.ada_augment = ada_augment

    def apply(self, x):
        if self.ada_p > 0:
            mask = torch.rand(x.size(0), device=DEVICE) < self.ada_p
            augmented = torch.stack([self.ada_augment(img) for img in x[mask]])
            x[mask] = augmented
        return x

    def update(self, logits):
        # Atualiza a probabilidade de augmentation
        ada_signs = torch.sign(logits).mean()
        self.ada_r_t = (1 - self.ada_speed) * self.ada_r_t + self.ada_speed * ada_signs
        self.ada_p = min(1.0, self.ada_p + self.ada_speed * (self.ada_r_t - self.ada_target))
        return self.ada_p

# -----------------------
# Inicialização dos modelos
# -----------------------
gen_in, disc_in = get_input_dimensions(Z_DIM, N_CLASSES, im_chan=1)

gen = Generator(Z_DIM + N_CLASSES, W_DIM, N_CLASSES,
                out_chan=1, base_ch=32, target_size=IMG_SIZE).to(DEVICE)
disc = Discriminator(1, N_CLASSES, base_ch=32, target_size=IMG_SIZE).to(DEVICE)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

gen.apply(weights_init)
disc.apply(weights_init)

opt_g = torch.optim.Adam(gen.parameters(), lr=LR_G, betas=BETAS)
opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=BETAS)

# -----------------------
# Checkpoint (resume auto)
# -----------------------
RESUME = True
CKPT_PATH = '/content/drive/MyDrive/training/checkpoints/ckpt_latest.pth'

start_epoch = 1
gen_losses, disc_losses, ada_ps = [], [], []

if RESUME and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    gen.load_state_dict(ckpt['gen'])
    disc.load_state_dict(ckpt['disc'])
    opt_g.load_state_dict(ckpt['gen_opt'])
    opt_d.load_state_dict(ckpt['disc_opt'])
    gen_losses = ckpt['gen_losses']
    disc_losses = ckpt['disc_losses']
    ada_ps = ckpt['ada_ps']
    start_epoch = ckpt['epoch'] + 1
    print(f"Retomando do epoch {start_epoch}")

# -----------------------
# Funções de loss
# -----------------------
def d_logistic_loss(real_pred, fake_pred):
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()

def g_logistic_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=False,
        retain_graph=False
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return R1_GAMMA / 2 * grad_penalty

# -----------------------
# Loop de treino
# -----------------------
fixed_noise = torch.randn(16, Z_DIM, device=DEVICE)
fixed_labels = torch.tensor([i % N_CLASSES for i in range(16)], device=DEVICE).long()
fixed_onehot = get_one_hot_labels(fixed_labels, N_CLASSES).to(DEVICE)
fixed_zc = combine_vectors(fixed_noise, fixed_onehot)

scaler_g = GradScaler()
scaler_d = GradScaler()

for epoch in range(start_epoch, N_EPOCHS + 1):
    pbar = tqdm(loader, desc=f'Época {epoch}/{N_EPOCHS}')
    for i, (real, labels) in enumerate(pbar):
        real = real.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        bs = real.size(0)

        noise = torch.randn(bs, Z_DIM, device=DEVICE)
        one_hot = get_one_hot_labels(labels, N_CLASSES).to(DEVICE)
        zc = combine_vectors(noise, one_hot)

        # ----------------- Discriminator -----------------
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = gen(zc)

        real_in = combine_vectors(real, one_hot[:, :, None, None].repeat(1, 1, IMG_SIZE, IMG_SIZE))
        fake_in = combine_vectors(fake.detach(), one_hot[:, :, None, None].repeat(1, 1, IMG_SIZE, IMG_SIZE))

        with autocast():
            real_pred = disc(real_in)
            fake_pred = disc(fake_in)
            d_loss = d_logistic_loss(real_pred, fake_pred)

        if i % 32 == 0:  # R1 menos frequente
            real_in_reg = real_in.detach().requires_grad_(True)
            with autocast():
                real_pred_reg = disc(real_in_reg)
                r1_loss = d_r1_loss(real_pred_reg, real_in_reg)
                d_loss = d_loss + r1_loss

        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # ----------------- Generator -----------------
        opt_g.zero_grad(set_to_none=True)
        with autocast():
            fake = gen(zc)
            fake_in = combine_vectors(fake, one_hot[:, :, None, None].repeat(1, 1, IMG_SIZE, IMG_SIZE))
            fake_pred = disc(fake_in)
            g_loss = g_logistic_loss(fake_pred)

        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        gen_losses.append(g_loss.item())
        disc_losses.append(d_loss.item())
        pbar.set_postfix({'g_loss': f'{g_loss.item():.3f}', 'd_loss': f'{d_loss.item():.3f}'})

    # ----------------- Checkpoint e imagens no Drive -----------------
    if epoch % 2 == 0:  # salvar a cada 2 épocas
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_zc).cpu()
            grid = make_grid((samples + 1) / 2, nrow=4, padding=2)
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
            plt.axis('off')
            plt.title(f'Amostras - Época {epoch}')
            plt.savefig(f'/content/drive/MyDrive/training/progress/epoch_{epoch}.png', bbox_inches='tight', dpi=150)
            plt.close()
        gen.train()

        torch.save({
            'epoch': epoch,
            'gen': gen.state_dict(),
            'disc': disc.state_dict(),
            'gen_opt': opt_g.state_dict(),
            'disc_opt': opt_d.state_dict(),
            'gen_losses': gen_losses,
            'disc_losses': disc_losses,
            'ada_ps': ada_ps,
        }, '/content/drive/MyDrive/training/checkpoints/ckpt_latest.pth')

print("Treinamento concluído!")