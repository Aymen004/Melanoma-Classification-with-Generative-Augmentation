"""
CALIBRAGE MASSIF VAE - Détermination du Seuil Optimal
======================================================

Ce script calcule les scores de reconstruction sur l'ensemble du dataset de calibrage
et génère les visualisations pour déterminer le seuil optimal de détection d'anomalies.

Visualisations générées:
1. Histogramme de distribution des scores (benign vs malignant)
2. Courbe ROC (True Positive Rate vs False Positive Rate)
3. Statistiques et seuil recommandé
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import argparse

from VAE_model import ConvVAE, VAEConfig


def load_model(checkpoint_path, latent_dim=512):
    """Charge le modèle VAE depuis un checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Créer la configuration
    config = VAEConfig(latent_dim=latent_dim)
    
    # Créer le modèle
    model = ConvVAE(config=config).to(device)
    
    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Modèle chargé depuis: {checkpoint_path}")
    print(f"   Device: {device}")
    return model, device


def load_labels(labels_csv):
    """Charge le fichier labels.csv et retourne un dictionnaire filename -> label."""
    df = pd.read_csv(labels_csv)
    labels_dict = {}
    
    for _, row in df.iterrows():
        filename = row['id']
        label = row['label']
        labels_dict[filename] = label
    
    print(f"✅ Labels chargés: {len(labels_dict)} images")
    print(f"   - Benign: {sum(1 for v in labels_dict.values() if v == 'benign')}")
    print(f"   - Malignant: {sum(1 for v in labels_dict.values() if v == 'malignant')}")
    
    return labels_dict


def get_reconstruction_score(model, img_path, transform, device, loss_type='l1'):
    """Calcule le score de reconstruction pour une image."""
    try:
        # Charger et transformer l'image
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        
        # Passer dans le VAE
        with torch.no_grad():
            recon, mu, logvar = model(x)
            
            # Calculer la perte de reconstruction
            if loss_type == 'l1':
                loss = F.l1_loss(recon, x, reduction='mean')
            else:  # mse
                loss = F.mse_loss(recon, x, reduction='mean')
        
        return loss.item()
    
    except Exception as e:
        print(f"   ⚠️  Erreur sur {img_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Calibrage massif VAE et visualisations')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du VAE')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Dossier contenant les images')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Fichier CSV contenant les labels')
    parser.add_argument('--output_dir', type=str, default='calibration_results',
                        help='Dossier de sortie pour les visualisations')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Dimension de l\'espace latent')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mse'],
                        help='Type de loss à calculer')
    args = parser.parse_args()
    
    # Créer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Transformation (même que l'entraînement)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Charger le modèle et les labels
    print("\n" + "="*80)
    print("🔧 CHARGEMENT DU MODÈLE ET DES DONNÉES")
    print("="*80)
    
    model, device = load_model(args.checkpoint, args.latent_dim)
    labels_dict = load_labels(args.labels_csv)
    
    # Calculer les scores pour toutes les images
    print("\n" + "="*80)
    print(f"📊 CALCUL DES SCORES DE RECONSTRUCTION ({args.loss_type.upper()} Loss)")
    print("="*80)
    
    results = []
    img_dir = Path(args.img_dir)
    
    # Trouver toutes les images
    all_images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"   Images trouvées: {len(all_images)}")
    
    # Calculer les scores avec barre de progression
    for img_path in tqdm(all_images, desc="Calcul des scores"):
        filename = img_path.name
        
        # Récupérer le label
        if filename not in labels_dict:
            continue
        
        label_str = labels_dict[filename]
        label_binary = 1 if label_str == 'malignant' else 0
        
        # Calculer le score
        score = get_reconstruction_score(model, img_path, transform, device, args.loss_type)
        
        if score is not None:
            results.append({
                'filename': filename,
                'score': score,
                'label': label_binary,
                'class': label_str
            })
    
    df = pd.DataFrame(results)
    
    # Sauvegarder les résultats
    csv_path = output_dir / 'scores_detailles.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Scores sauvegardés: {csv_path}")
    print(f"   Total: {len(df)} images")
    
    # --- STATISTIQUES ---
    print("\n" + "="*80)
    print("📊 STATISTIQUES DES SCORES")
    print("="*80)
    
    benign_scores = df[df['class'] == 'benign']['score']
    malignant_scores = df[df['class'] == 'malignant']['score']
    
    print(f"\n🟢 BENIGN (n={len(benign_scores)}):")
    print(f"   Moyenne: {benign_scores.mean():.6f}")
    print(f"   Médiane: {benign_scores.median():.6f}")
    print(f"   Std Dev: {benign_scores.std():.6f}")
    print(f"   Min-Max: {benign_scores.min():.6f} - {benign_scores.max():.6f}")
    print(f"   Percentile 95%: {benign_scores.quantile(0.95):.6f}")
    
    print(f"\n🔴 MALIGNANT (n={len(malignant_scores)}):")
    print(f"   Moyenne: {malignant_scores.mean():.6f}")
    print(f"   Médiane: {malignant_scores.median():.6f}")
    print(f"   Std Dev: {malignant_scores.std():.6f}")
    print(f"   Min-Max: {malignant_scores.min():.6f} - {malignant_scores.max():.6f}")
    print(f"   Percentile 5%: {malignant_scores.quantile(0.05):.6f}")
    
    separation_ratio = malignant_scores.mean() / benign_scores.mean()
    print(f"\n📊 Ratio Malignant/Benign: {separation_ratio:.2f}x")
    
    # --- VISUALISATION 1 : HISTOGRAMME ---
    print("\n" + "="*80)
    print("📈 GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    
    plt.figure(figsize=(12, 6))
    
    # Utiliser des bins adaptés
    bins = np.linspace(df['score'].min(), df['score'].max(), 40)
    
    plt.hist(benign_scores, bins=bins, alpha=0.6, label='Benign', color='green', edgecolor='black')
    plt.hist(malignant_scores, bins=bins, alpha=0.6, label='Malignant', color='red', edgecolor='black')
    
    plt.axvline(benign_scores.mean(), color='green', linestyle='--', linewidth=2, label=f'Benign Mean ({benign_scores.mean():.4f})')
    plt.axvline(malignant_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Malignant Mean ({malignant_scores.mean():.4f})')
    plt.axvline(benign_scores.quantile(0.95), color='darkgreen', linestyle=':', linewidth=2, label=f'95% Benign ({benign_scores.quantile(0.95):.4f})')
    
    plt.xlabel('Reconstruction Score (L1 Loss)', fontsize=12)
    plt.ylabel('Nombre d\'images', fontsize=12)
    plt.title('Distribution des Scores d\'Anomalie - VAE Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    hist_path = output_dir / 'calibration_histogram.png'
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"✅ Histogramme sauvegardé: {hist_path}")
    
    # --- VISUALISATION 2 : ROC CURVE ---
    fpr, tpr, thresholds = roc_curve(df['label'], df['score'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Fausses Alertes)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensibilité / Rappel)', fontsize=12)
    plt.title('Courbe ROC - Performance de Détection d\'Anomalies', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = output_dir / 'calibration_roc.png'
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"✅ Courbe ROC sauvegardée: {roc_path}")
    
    # --- VISUALISATION 3 : PRECISION-RECALL CURVE ---
    precision, recall, pr_thresholds = precision_recall_curve(df['label'], df['score'])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, color='purple', lw=3, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensibilité)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Courbe Précision-Rappel', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pr_path = output_dir / 'calibration_precision_recall.png'
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"✅ Courbe Précision-Rappel sauvegardée: {pr_path}")
    
    # --- TROUVER LE MEILLEUR SEUIL ---
    print("\n" + "="*80)
    print("🎯 DÉTERMINATION DU SEUIL OPTIMAL")
    print("="*80)
    
    # Stratégie 1: Seuil pour 90% de rappel (ne pas rater les cancers)
    if np.any(tpr >= 0.9):
        idx_90_recall = np.where(tpr >= 0.9)[0][0]
        threshold_90_recall = thresholds[idx_90_recall]
        fpr_90_recall = fpr[idx_90_recall]
        tpr_90_recall = tpr[idx_90_recall]
        
        print(f"\n📌 SEUIL pour 90% de Rappel (Sensibilité):")
        print(f"   Threshold: {threshold_90_recall:.6f}")
        print(f"   True Positive Rate: {tpr_90_recall*100:.1f}%")
        print(f"   False Positive Rate: {fpr_90_recall*100:.1f}%")
        print(f"   ➡️  Détecte {tpr_90_recall*100:.1f}% des cancers, mais {fpr_90_recall*100:.1f}% de fausses alertes")
    
    # Stratégie 2: Seuil pour 95% de rappel (très sensible)
    if np.any(tpr >= 0.95):
        idx_95_recall = np.where(tpr >= 0.95)[0][0]
        threshold_95_recall = thresholds[idx_95_recall]
        fpr_95_recall = fpr[idx_95_recall]
        tpr_95_recall = tpr[idx_95_recall]
        
        print(f"\n📌 SEUIL pour 95% de Rappel (Très Sensible):")
        print(f"   Threshold: {threshold_95_recall:.6f}")
        print(f"   True Positive Rate: {tpr_95_recall*100:.1f}%")
        print(f"   False Positive Rate: {fpr_95_recall*100:.1f}%")
        print(f"   ➡️  Détecte {tpr_95_recall*100:.1f}% des cancers, mais {fpr_95_recall*100:.1f}% de fausses alertes")
    
    # Stratégie 3: Seuil de Youden (maximise TPR - FPR)
    youden_index = tpr - fpr
    idx_youden = np.argmax(youden_index)
    threshold_youden = thresholds[idx_youden]
    tpr_youden = tpr[idx_youden]
    fpr_youden = fpr[idx_youden]
    
    print(f"\n📌 SEUIL de Youden (Balance Optimale):")
    print(f"   Threshold: {threshold_youden:.6f}")
    print(f"   True Positive Rate: {tpr_youden*100:.1f}%")
    print(f"   False Positive Rate: {fpr_youden*100:.1f}%")
    print(f"   Youden Index: {youden_index[idx_youden]:.3f}")
    
    # Stratégie 4: Basé sur percentile 95 des bénins
    threshold_p95 = benign_scores.quantile(0.95)
    predictions_p95 = (df['score'] > threshold_p95).astype(int)
    tp_p95 = ((predictions_p95 == 1) & (df['label'] == 1)).sum()
    fp_p95 = ((predictions_p95 == 1) & (df['label'] == 0)).sum()
    tn_p95 = ((predictions_p95 == 0) & (df['label'] == 0)).sum()
    fn_p95 = ((predictions_p95 == 0) & (df['label'] == 1)).sum()
    
    tpr_p95 = tp_p95 / (tp_p95 + fn_p95) if (tp_p95 + fn_p95) > 0 else 0
    fpr_p95 = fp_p95 / (fp_p95 + tn_p95) if (fp_p95 + tn_p95) > 0 else 0
    
    print(f"\n📌 SEUIL au Percentile 95% des Bénins:")
    print(f"   Threshold: {threshold_p95:.6f}")
    print(f"   True Positive Rate: {tpr_p95*100:.1f}%")
    print(f"   False Positive Rate: {fpr_p95*100:.1f}%")
    
    # --- RÉSUMÉ FINAL ---
    print("\n" + "="*80)
    print("📋 RÉSUMÉ FINAL")
    print("="*80)
    
    print(f"\n🏆 Score AUC-ROC: {roc_auc:.4f}")
    if roc_auc >= 0.9:
        print("   ➡️  EXCELLENT - Le modèle sépare très bien benign/malignant")
    elif roc_auc >= 0.8:
        print("   ➡️  TRÈS BON - Le modèle est performant")
    elif roc_auc >= 0.7:
        print("   ➡️  BON - Le modèle fonctionne correctement")
    elif roc_auc >= 0.6:
        print("   ➡️  MOYEN - Le modèle a du potentiel mais peut être amélioré")
    else:
        print("   ➡️  FAIBLE - Le modèle nécessite des améliorations")
    
    print(f"\n🎯 SEUIL RECOMMANDÉ pour usage médical:")
    print(f"   Threshold: {threshold_youden:.6f}")
    print(f"   (Compromis optimal entre sensibilité et spécificité)")
    
    print("\n" + "="*80)
    print("✅ CALIBRAGE TERMINÉ")
    print("="*80)
    print(f"\nFichiers générés dans: {output_dir.absolute()}")
    print(f"   - {hist_path.name}")
    print(f"   - {roc_path.name}")
    print(f"   - {pr_path.name}")
    print(f"   - {csv_path.name}")
    print()


if __name__ == '__main__':
    main()
