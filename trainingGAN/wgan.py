
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import matplotlib
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
from torch import nn
# matplot não exibe gráficos interativos
matplotlib.use('Agg')
# resultados reproduzíveis
torch.manual_seed(0)

class CustomMedicalDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)  # carrega csv
        self.root_dir = root_dir  # diretório imgs
        self.transform = transform  # transformações que serão aplicadas
        # label (0 = BENIGNO, 1 = MALIGNO)
        self.data['label'] = self.data['pathology'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    # qtde de imagens
    def __len__(self):
        return len(self.data)
    
    # processa imagem
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['image file path'] + '.jpg'
        img_path = os.path.join(self.root_dir, filename)

        # carrega imagem em escala de cinza
        image = Image.open(img_path).convert('L')
        
        # aplica transformações se especificado
        if self.transform:
            image = self.transform(image)

        # converte rótulo para tensor
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label
    
class Generator(nn.Module):
    def __init__(self, input_dim, im_chan=1, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            # 1×1 → 4×4
            self.make_gen_block(input_dim,   hidden_dim*16, kernel_size=4, stride=1, padding=0),
            # 4×4 → 8×8
            self.make_gen_block(hidden_dim*16, hidden_dim*8),
            # 8×8 → 16×16
            self.make_gen_block(hidden_dim*8,  hidden_dim*4),
            # 16×16 → 32×32
            self.make_gen_block(hidden_dim*4,  hidden_dim*2),
            # 32×32 → 64×64
            self.make_gen_block(hidden_dim*2,  hidden_dim),
            # 64×64 → 128×128
            self.make_gen_block(hidden_dim,    hidden_dim//2),
            # 128×128 → 256×256
            self.make_gen_block(hidden_dim//2, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True),
        )

    # cria um bloco do gerador
    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),  # saída entre -1 e 1
            )

    # passa o ruído através da rede
    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super().__init__()
        self.disc = nn.Sequential(
            # 256 → 128
            self.make_disc_block(im_chan,      hidden_dim//2, kernel_size=4, stride=2, padding=1),
            # 128 → 64
            self.make_disc_block(hidden_dim//2, hidden_dim,    kernel_size=4, stride=2, padding=1),
            # 64 → 32
            self.make_disc_block(hidden_dim,    hidden_dim*2),
            # 32 → 16
            self.make_disc_block(hidden_dim*2,  hidden_dim*4),
            # 16 → 8
            self.make_disc_block(hidden_dim*4,  hidden_dim*8),
            # 8 → 4
            self.make_disc_block(hidden_dim*8,  hidden_dim*16),
            # 4 → 1
            self.make_disc_block(hidden_dim*16, 1,            kernel_size=4, stride=1, padding=0, final_layer=True),
        )

    # cria um bloco do discriminador
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3),  # regularização
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                # sem ativação para usar com loss BCEWithLogits
            )

    # passa a imagem através da rede
    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
    

# tamanho das imagens de saída
img_size = 256
# transformações aplicadas às imagens
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Redimensiona
    transforms.RandomRotation(15),            # Rotação aleatória
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variação de cor
    transforms.ToTensor(),                    # Converte para tensor
    transforms.Normalize((0.5,), (0.5,)),     # Normaliza entre -1 e 1
])

# parâmetros da GAN
n_classes = 2          # Número de classes (benigno/maligno)
img_shape = (1, img_size, img_size)  # Formato das imagens (C, H, W)
z_dim = 100            # Dimensão do vetor latente (ruído de entrada)


# converte rótulos para one-hot encoding
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

# combina dois vetores para entrada condicional
def combine_vectors(x, y):
    return torch.cat((x.float(), y.float()), 1)

# calcula dimensões de entrada para gerador e discriminador
def get_input_dimensions(z_dim, img_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = img_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan

# calcula o gradient penalty para WGAN-GP
def gradient_penalty(disc, real, fake, labels, device):
    batch_size = real.shape[0]
    # gera pontos interpolados
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    
    # prepara os rótulos
    one_hot_labels = get_one_hot_labels(labels, n_classes)
    image_one_hot_labels = one_hot_labels[:, :, None, None]
    image_one_hot_labels = image_one_hot_labels.repeat(1, 1, *img_shape[1:])
    interpolated_and_labels = combine_vectors(interpolated, image_one_hot_labels)
    
    # calcula a saída do discriminador
    disc_interpolated = disc(interpolated_and_labels)
    
    # calcula gradientes
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # calcula a penalidade
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty

# inicializa os pesos com distribuição normal
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


# configuração de treinamento
csv_path = r'/home/rptokairin/progs/tcc/training/dataset.csv'
root_dir = r'/home/rptokairin/progs/tcc/training/dataset'

# cria dataset
dataset = CustomMedicalDataset(csv_path, root_dir, transform=transform)
print(f"Tamanho do dataset: {len(dataset)} imagens")

# cria dataloader
batch_size = 16  # Tamanho do batch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# calcula dimensões de entrada
generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, img_shape, n_classes)

# instancia modelos
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)

# inicialização de pesos
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

# configura otimizadores
lr = 0.0001  # taxa de aprendizado
weight_decay = 1e-5  # decaimento de peso para regularização
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

# parâmetros WGAN-GP
lambda_gp = 10  # peso do gradient penalty
n_critic = 5    # número de iterações do discriminador por iteração do gerador
n_epochs = 500  # número total de épocas

# prepara diretório para salvar resultados
os.makedirs("progress", exist_ok=True)

# listas para armazenar as perdas
gen_losses = []
disc_losses = []

# treinamento
for epoch in range(n_epochs):
    for real, labels in tqdm(dataloader, desc=f"Época {epoch+1}/{n_epochs}"):
        cur_batch_size = len(real)
        real = real.to(device)
        labels = labels.to(device)
        
        # treino discriminador
        for _ in range(n_critic):
            disc_opt.zero_grad()
            
            # gera imagens falsas
            fake_noise = torch.randn(cur_batch_size, z_dim, device=device)
            one_hot_labels = get_one_hot_labels(labels, n_classes)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            
            # prepara entradas com rótulos
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, *img_shape[1:])
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            
            # calcula as saídas
            disc_fake_pred = disc(fake_image_and_labels.detach())
            disc_real_pred = disc(real_image_and_labels)
            
            # calcula gradient penalty
            gp = gradient_penalty(disc, real, fake, labels, device)
            
            # calcula a loss total
            disc_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + lambda_gp * gp
            disc_loss.backward()
            disc_opt.step()
        
        # treino gerador
        gen_opt.zero_grad()
        fake_noise = torch.randn(cur_batch_size, z_dim, device=device)
        one_hot_labels = get_one_hot_labels(labels, n_classes)
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)
        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        
        # loss do gerador (tenta enganar o discriminador)
        gen_loss = -torch.mean(disc_fake_pred)
        gen_loss.backward()
        gen_opt.step()
        
        # armazena as perdas
        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

    if (epoch + 1) % 10 == 0:
        # gera imagens de exemplo
        with torch.no_grad():
            fake_noise = torch.randn(16, z_dim, device=device)
            random_labels = torch.randint(0, n_classes, (16,), device=device)
            one_hot_labels = get_one_hot_labels(random_labels, n_classes)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            examples = gen(noise_and_labels)
            
            # salva as imagens
            plt.figure(figsize=(8,8))
            grid = make_grid((examples + 1) / 2, nrow=4)
            plt.imshow(grid.permute(1, 2, 0).cpu().squeeze(), cmap='gray')
            plt.axis("off")
            plt.title(f"Época {epoch+1}")
            plt.savefig(f"progress/epoch_{epoch+1}.png")
            plt.close()
            print(f"\nImagens salvas em: progress/epoch_{epoch+1}.png")

        # exibe métricas
        current_gen_loss = sum(gen_losses[-len(dataloader):])/len(dataloader)
        current_disc_loss = sum(disc_losses[-len(dataloader):])/len(dataloader)
        
        print(f"\nÉpoca {epoch+1}/{n_epochs}")
        print(f"Perda do Gerador: {current_gen_loss:.4f}")
        print(f"Perda do Discriminador: {current_disc_loss:.4f}")
        
        # limpa a memória da GPU
        torch.cuda.empty_cache()

        # salva checkpoint do modelo
        torch.save({
            'gen': gen.state_dict(),
            'disc': disc.state_dict(),
            'gen_opt': gen_opt.state_dict(),
            'disc_opt': disc_opt.state_dict(),
            'gen_losses': gen_losses,
            'disc_losses': disc_losses,
        }, f'checkpoint_epoch_{epoch+1}.pth')

print("Treinamento concluído!")