# =====================================================
# Avaliação de GAN com LPIPS (cross-set) em mamografias
# =====================================================
import csv
import os
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
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
# Quantidade de sintéticas a serem geradas para avaliação
TARGET_IMAGES = {
    0: 1000,  # Benignas
    1: 1000   # Malignas
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

# =====================================================
# FUNÇÕES PARA ANÁLISE DE FREQUÊNCIA LPIPS
# =====================================================
def plot_lpips_frequency(lpips_scores_benign, lpips_scores_malignant, output_dir="lpips_analysis"):
    """Plota gráficos de frequência dos valores LPIPS"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combinar todos os scores para análise geral
    all_scores = lpips_scores_benign + lpips_scores_malignant
    
    # Configuração dos gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histograma combinado
    axes[0, 0].hist(all_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('Valor LPIPS')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].set_title('Distribuição de Valores LPIPS - Todas as Imagens')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Adicionar linhas verticais para estatísticas
    mean_all = np.mean(all_scores)
    median_all = np.median(all_scores)
    axes[0, 0].axvline(mean_all, color='red', linestyle='--', label=f'Média: {mean_all:.3f}')
    axes[0, 0].axvline(median_all, color='orange', linestyle='--', label=f'Mediana: {median_all:.3f}')
    axes[0, 0].legend()
    
    # 2. Histogramas separados por classe
    axes[0, 1].hist(lpips_scores_benign, bins=50, alpha=0.7, color='blue', 
                   label=f'Benign (n={len(lpips_scores_benign)})', edgecolor='black')
    axes[0, 1].hist(lpips_scores_malignant, bins=50, alpha=0.7, color='red', 
                   label=f'Malignant (n={len(lpips_scores_malignant)})', edgecolor='black')
    axes[0, 1].set_xlabel('Valor LPIPS')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição de Valores LPIPS por Classe')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gráfico de densidade (KDE)
    from scipy import stats
    x_range = np.linspace(min(all_scores), max(all_scores), 100)
    
    if len(lpips_scores_benign) > 1:
        kde_benign = stats.gaussian_kde(lpips_scores_benign)
        axes[1, 0].plot(x_range, kde_benign(x_range), color='blue', label='Benign', linewidth=2)
    
    if len(lpips_scores_malignant) > 1:
        kde_malignant = stats.gaussian_kde(lpips_scores_malignant)
        axes[1, 0].plot(x_range, kde_malignant(x_range), color='red', label='Malignant', linewidth=2)
    
    axes[1, 0].set_xlabel('Valor LPIPS')
    axes[1, 0].set_ylabel('Densidade')
    axes[1, 0].set_title('Densidade de Probabilidade LPIPS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot comparativo
    box_data = [lpips_scores_benign, lpips_scores_malignant]
    box_labels = ['Benign', 'Malignant']
    box_plot = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Cores para os box plots
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1, 1].set_ylabel('Valor LPIPS')
    axes[1, 1].set_title('Distribuição LPIPS - Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lpips_frequency_analysis.png'), dpi=300)
    plt.close()
    
    # Gráfico de dispersão individual
    plt.figure(figsize=(12, 6))
    
    # Plot para Benign
    plt.scatter(range(len(lpips_scores_benign)), lpips_scores_benign, 
               alpha=0.6, label=f'Benign (n={len(lpips_scores_benign)})', 
               color='blue', s=30)
    
    # Plot para Malignant
    plt.scatter(range(len(lpips_scores_malignant)), lpips_scores_malignant, 
               alpha=0.6, label=f'Malignant (n={len(lpips_scores_malignant)})', 
               color='red', s=30)
    
    plt.xlabel('Índice da Imagem Gerada')
    plt.ylabel('Menor Valor LPIPS')
    plt.title('Distribuição dos Menores Valores LPIPS por Imagem Gerada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar linhas horizontais de referência
    reference_lines = [0.1, 0.2, 0.3, 0.4, 0.5]
    colors_ref = ['green', 'orange', 'red', 'purple', 'brown']
    for ref, color in zip(reference_lines, colors_ref):
        plt.axhline(y=ref, color=color, linestyle='--', alpha=0.5, label=f'LPIPS = {ref}')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lpips_scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvar estatísticas detalhadas
    stats = {
        'overall': {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'median': float(np.median(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'q1': float(np.percentile(all_scores, 25)),
            'q3': float(np.percentile(all_scores, 75)),
            'count': len(all_scores)
        },
        'benign': {
            'mean': float(np.mean(lpips_scores_benign)),
            'std': float(np.std(lpips_scores_benign)),
            'median': float(np.median(lpips_scores_benign)),
            'min': float(np.min(lpips_scores_benign)),
            'max': float(np.max(lpips_scores_benign)),
            'q1': float(np.percentile(lpips_scores_benign, 25)),
            'q3': float(np.percentile(lpips_scores_benign, 75)),
            'count': len(lpips_scores_benign)
        },
        'malignant': {
            'mean': float(np.mean(lpips_scores_malignant)),
            'std': float(np.std(lpips_scores_malignant)),
            'median': float(np.median(lpips_scores_malignant)),
            'min': float(np.min(lpips_scores_malignant)),
            'max': float(np.max(lpips_scores_malignant)),
            'q1': float(np.percentile(lpips_scores_malignant, 25)),
            'q3': float(np.percentile(lpips_scores_malignant, 75)),
            'count': len(lpips_scores_malignant)
        }
    }
    
    with open(os.path.join(output_dir, 'lpips_detailed_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print resumo estatístico
    print("\n" + "="*60)
    print("RESUMO ESTATÍSTICO LPIPS")
    print("="*60)
    print(f"Overall: {stats['overall']['mean']:.4f} ± {stats['overall']['std']:.4f}")
    print(f"Benign:  {stats['benign']['mean']:.4f} ± {stats['benign']['std']:.4f}")
    print(f"Malignant: {stats['malignant']['mean']:.4f} ± {stats['malignant']['std']:.4f}")
    print(f"Range: [{stats['overall']['min']:.4f}, {stats['overall']['max']:.4f}]")
    
    print(f"\nAnálise salva em: {output_dir}")

# =====================================================
# PIPELINE PRINCIPAL PARA AVALIAÇÃO
# =====================================================
def evaluate_lpips_distribution(gen, dataset_list, loss_fn, class_label, class_name, n_images):
    """Gera imagens e calcula distribuição LPIPS sem threshold"""
    real_images = load_real_images_from_csv(dataset_list, class_label, 100)
    if not real_images:
        print(f"Nenhuma imagem real disponível para {class_name}.")
        return []

    min_lpips_scores = []
    
    print(f"Gerando {n_images} imagens para análise LPIPS - {class_name}")
    for i in tqdm(range(n_images), desc=f"Gerando {class_name}"):
        # gerar uma imagem
        noise = torch.randn(1, Z_DIM, device=DEVICE)
        label = torch.tensor([class_label], dtype=torch.long, device=DEVICE)
        one_hot = get_one_hot_labels(label, N_CLASSES).to(DEVICE)
        zc = combine_vectors(noise, one_hot)
        with torch.no_grad():
            fake = gen(zc).cpu()

        # calcular LPIPS contra TODAS as imagens reais
        fake_rgb = fake.to(DEVICE).repeat(1, 3, 1, 1)
        scores = []
        for real in real_images:
            real_rgb = real.to(DEVICE).repeat(1, 3, 1, 1)
            with torch.no_grad():
                score = loss_fn(real_rgb, fake_rgb).item()
            scores.append(score)

        # Encontrar o menor valor LPIPS (mais similar)
        min_score = np.min(scores)
        min_lpips_scores.append(min_score)

    print(f"{class_name}: LPIPS médio = {np.mean(min_lpips_scores):.4f} ± {np.std(min_lpips_scores):.4f}")
    return min_lpips_scores

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

    # Avaliar distribuição LPIPS para ambas as classes
    lpips_scores_benign = []
    lpips_scores_malignant = []
    
    for class_label, class_name in enumerate(['BENIGN', 'MALIGNANT']):
        n_images = TARGET_IMAGES[class_label]
        
        if class_label == 0:
            lpips_scores_benign = evaluate_lpips_distribution(gen, dataset_list, loss_fn, 
                                                            class_label, class_name, n_images)
        else:
            lpips_scores_malignant = evaluate_lpips_distribution(gen, dataset_list, loss_fn, 
                                                               class_label, class_name, n_images)

    # Plotar análise de frequência
    print("\n=== Gerando análise de frequência LPIPS ===")
    plot_lpips_frequency(lpips_scores_benign, lpips_scores_malignant)

if __name__ == "__main__":
    main()