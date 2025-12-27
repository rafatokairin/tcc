import os
import argparse
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------
# Dataset definições
# ----------------------
class MammogramDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_real_dataset(csv_path: Path, real_root: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    label_map = {"BENIGN": 0, "MALIGNANT": 1}

    filepaths, labels = [], []
    for _, row in df.iterrows():
        raw_label = str(row["pathology"]).strip().upper()
        raw_path = str(row["image file path"]).strip()
        if raw_label not in label_map:
            continue
        candidate = real_root / raw_path
        if not candidate.exists():
            candidate = real_root / f"{raw_path}.jpg"
        if candidate.exists():
            filepaths.append(str(candidate))
            labels.append(label_map[raw_label])

    return pd.DataFrame({"filepath": filepaths, "label": labels})


def downsample_benign(df: pd.DataFrame, benign_target=245):
    benign_df = df[df.label == 0]
    malignant_df = df[df.label == 1]
    benign_sample = benign_df.sample(n=benign_target, random_state=SEED)
    reduced = pd.concat([benign_sample, malignant_df], axis=0).sample(frac=1, random_state=SEED)
    return reduced.reset_index(drop=True)

# ----------------------
# Training funções
# ----------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calcular métricas adicionais
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return running_loss / total, acc, f1, auc, cm

# ----------------------
# Cross-Validation rotina
# ----------------------

def run_cross_validation(args, real_data, n_splits=5):
    """Executa validação cruzada estratificada apenas com dados reais"""
    
    # Preparar transformações
    transform_train = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    transform_val = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurar K-Fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    fold_results = []
    best_models = []
    
    # Extrair features e labels para CV
    X = real_data['filepath'].values
    y = real_data['label'].values
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Dividir dados reais
        X_train_real, X_val_real = X[train_idx], X[val_idx]
        y_train_real, y_val_real = y[train_idx], y[val_idx]
        
        # Criar datasets (apenas dados reais)
        train_dataset = MammogramDataset(X_train_real.tolist(), y_train_real.tolist(), transform=transform_train)
        val_dataset = MammogramDataset(X_val_real.tolist(), y_val_real.tolist(), transform=transform_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Inicializar modelo
        model = models.efficientnet_b0(pretrained=True)

        # Treinar só a cabeça primeiro (opcional)
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # LR para cabeça

        best_fold_acc = 0.0
        # Número de épocas reduzido
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_auc, val_cm = evaluate_model(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{args.epochs} - "
                f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")

            
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_model = model.state_dict().copy()
        
        # Avaliação final do fold
        final_loss, final_acc, final_f1, final_auc, final_cm = evaluate_model(
            model, val_loader, criterion, device
        )
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': final_acc,
            'f1_score': final_f1,
            'auc': final_auc,
            'confusion_matrix': final_cm,
            'best_accuracy': best_fold_acc
        })
        
        best_models.append(best_fold_model)
        
        print(f"\nFold {fold + 1} Final Results:")
        print(f"Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}, AUC: {final_auc:.4f}")
        print(f"Confusion Matrix:\n{final_cm}")
    
    return fold_results, best_models

# ----------------------
# Main rotina
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", type=str, required=True)
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar dados
    print("Loading datasets...")
    real_df = load_real_dataset(Path(args.real_csv), Path(args.real_root))
    reduced_df = downsample_benign(real_df, benign_target=245)
    
    print(f"Real data: {len(reduced_df)} images")
    print(f"Class distribution - Real: {reduced_df['label'].value_counts().to_dict()}")

    # Executar validação cruzada
    print(f"\nStarting {args.folds}-fold Cross-Validation...")
    results, models = run_cross_validation(args, reduced_df, n_splits=args.folds)

    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "cross_validation_results.csv", index=False)
    
    # Calcular estatísticas finais
    final_metrics = {
        'mean_accuracy': np.mean([r['accuracy'] for r in results]),
        'std_accuracy': np.std([r['accuracy'] for r in results]),
        'mean_f1': np.mean([r['f1_score'] for r in results]),
        'std_f1': np.std([r['f1_score'] for r in results]),
        'mean_auc': np.mean([r['auc'] for r in results]),
        'std_auc': np.std([r['auc'] for r in results]),
    }
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {final_metrics['mean_accuracy']:.4f} ± {final_metrics['std_accuracy']:.4f}")
    print(f"Mean F1-Score: {final_metrics['mean_f1']:.4f} ± {final_metrics['std_f1']:.4f}")
    print(f"Mean AUC: {final_metrics['mean_auc']:.4f} ± {final_metrics['std_auc']:.4f}")
    
    # Salvar métricas finais
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Salvar melhor modelo (último fold)
    torch.save(models[-1], out_dir / "best_model_cv.pt")
    
    with open(out_dir / "class_index.json", "w") as f:
        json.dump({"BENIGN": 0, "MALIGNANT": 1}, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()