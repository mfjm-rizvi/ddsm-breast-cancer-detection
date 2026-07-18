"""
CM3015 Breast Cancer Detection: Shared ABMIL pipeline components
Used by NB03, NB04, NB05, NB07 (and any future backbone notebooks).
Identical logic to what was inline in NB03-05. This is a relocation,
not a rewrite. No prior results are affected by importing from here.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import timm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_normalisation_tensors(device):
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def build_backbone(model_name, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


class PatchDataset(Dataset):
    def __init__(self, X_np, y_np):
        self.X = X_np
        self.y = torch.from_numpy(y_np).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.X[idx]).float()
        return patch.permute(2, 0, 1), self.y[idx]


class PatchClassifier(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class AttentionPool(nn.Module):
    def __init__(self, feat_dim, attn_dim, dropout=0.0):
        super().__init__()
        self.V = nn.Linear(feat_dim, attn_dim, bias=True)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        a = torch.softmax(self.w(self.dropout(torch.tanh(self.V(h)))), dim=0)
        z = torch.sum(a * h, dim=0, keepdim=True)
        return z, a


class BagClassifier(nn.Module):
    def __init__(self, feat_dim, attn_dim, dropout=0.0):
        super().__init__()
        self.attn = AttentionPool(feat_dim, attn_dim, dropout)
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, h):
        z, a = self.attn(h)
        return self.head(z).squeeze(), a


class CachedBagDataset(Dataset):
    def __init__(self, features, y_np, bag_ids, bag_list):
        self.features, self.y, self.bag_ids, self.bag_list = features, y_np, bag_ids, bag_list

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, idx):
        bid = self.bag_list[idx]
        mask = self.bag_ids == bid
        h = torch.from_numpy(self.features[mask]).float()
        bag_label = torch.tensor(self.y[mask][0], dtype=torch.float32)
        return h, bag_label


def collate_cached(batch):
    return [item[0] for item in batch], torch.stack([item[1] for item in batch])


def extract_features(X_np, model, mean_gpu, std_gpu, device, batch_size=256):
    all_feats = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            batch_np = X_np[start:start + batch_size]
            t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            t = t.repeat(1, 3, 1, 1)
            t = (t - mean_gpu) / std_gpu
            with torch.amp.autocast('cuda'):
                feats = model(t).cpu().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def compute_all_metrics(y_true, y_probs, threshold, label=""):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_val     = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10, strategy='uniform')
    bin_counts = np.histogram(y_probs, bins=np.linspace(0, 1, 11))[0]
    ece = float(np.sum(bin_counts[:len(prob_true)] * np.abs(prob_true - prob_pred)) / len(y_probs))

    metrics = {
        "threshold": threshold,
        "auc": float(roc_auc_score(y_true, y_probs)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr_val),
        "ece": ece,
    }
    if label:
        print(f"── {label} ──")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")
    return metrics