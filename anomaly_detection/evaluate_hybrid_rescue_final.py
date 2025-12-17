"""
Évaluation du Système Hybride : DenseNet + VAE (Mode Rescue)

Logique :
- Si DenseNet dit MALIGNANT (prob >= thresh_densenet) → GARDER MALIGNANT
- Si DenseNet dit BENIGN (prob < thresh_densenet) :
    - Si VAE détecte anomalie (score >= thresh_vae) → FORCER MALIGNANT (RESCUE!)
    - Sinon → GARDER BENIGN

Objectif : Réduire les Faux Négatifs (FN) - Sauver les cancers manqués
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# Ajouter les chemins
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classifiers'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from densenet121 import DenseNet121Classifier
from VAE_model import ConvVAE, VAEConfig


class BinaryImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = ['benign', 'malignant']
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx, img_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, img_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_name


def load_densenet(checkpoint_path, device):
    """Charge le modèle DenseNet"""
    print(f"\n🔧 Chargement DenseNet: {checkpoint_path}")
    model = DenseNet121Classifier(num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remapper les clés
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('densenet.'):
            new_key = key.replace('densenet.', 'features.')
            new_state_dict[new_key] = value
        elif key.startswith('backbone.'):
            new_key = key.replace('backbone.', 'features.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✓ DenseNet chargé")
    return model


def load_vae(checkpoint_path, latent_dim, device):
    """Charge le modèle VAE"""
    print(f"\n🔧 Chargement VAE: {checkpoint_path}")
    config = VAEConfig(latent_dim=latent_dim, image_size=128)
    model = ConvVAE(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ VAE chargé")
    return model


def evaluate_hybrid_system(densenet, vae, dataloader, device, 
                          thresh_densenet=0.3, thresh_vae=0.136):
    """
    Évalue le système hybride avec mode rescue
    """
    densenet.eval()
    vae.eval()
    
    all_labels = []
    all_densenet_probs = []
    all_vae_scores = []
    all_filenames = []
    
    print(f"\n🔄 Évaluation du système hybride...")
    print(f"   Seuil DenseNet: {thresh_densenet}")
    print(f"   Seuil VAE: {thresh_vae}")
    
    # Transform pour redimensionner pour le VAE (128x128)
    resize_vae = transforms.Resize((128, 128))
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Processing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. Prédictions DenseNet (images sont déjà 224x224)
            outputs = densenet(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob malignant
            
            # 2. Scores VAE (besoin de redimensionner à 128x128)
            images_vae = resize_vae(images)
            recon, _, _ = vae(images_vae)
            vae_scores = torch.nn.functional.l1_loss(recon, images_vae, reduction='none')
            vae_scores = vae_scores.view(vae_scores.size(0), -1).mean(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_densenet_probs.extend(probs.cpu().numpy())
            all_vae_scores.extend(vae_scores.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Convertir en arrays
    y_true = np.array(all_labels)
    densenet_probs = np.array(all_densenet_probs)
    vae_scores = np.array(all_vae_scores)
    
    # --- BASELINE : DenseNet seul ---
    y_pred_densenet = (densenet_probs >= thresh_densenet).astype(int)
    
    # --- HYBRIDE : DenseNet + VAE Rescue ---
    y_pred_hybrid = y_pred_densenet.copy()
    
    # RESCUE LOGIC : Si DenseNet dit "Bénin" MAIS VAE détecte anomalie
    mask_rescue = (y_pred_densenet == 0) & (vae_scores >= thresh_vae)
    y_pred_hybrid[mask_rescue] = 1  # Forcer à MALIGNANT
    
    n_rescued = np.sum(mask_rescue)
    print(f"\n🔍 Le VAE a modifié {n_rescued} diagnostics (Bénin → Malin/Suspect)")
    
    # Créer DataFrame pour analyse détaillée
    df_results = pd.DataFrame({
        'filename': all_filenames,
        'label': y_true,
        'densenet_prob': densenet_probs,
        'vae_score': vae_scores,
        'pred_densenet': y_pred_densenet,
        'pred_hybrid': y_pred_hybrid,
        'rescued': mask_rescue
    })
    
    return df_results, y_pred_densenet, y_pred_hybrid


def calculate_metrics(y_true, y_pred, model_name):
    """Calcule toutes les métriques"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Model': model_name,
        'Accuracy': accuracy * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'Specificity': specificity * 100,
        'F1-Score': f1 * 100,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


def generate_comparison_report(metrics_base, metrics_hybrid, output_dir):
    """Génère un rapport comparatif détaillé"""
    
    print("\n" + "="*80)
    print("📊 RÉSULTATS COMPARATIFS")
    print("="*80)
    
    # DataFrame pour affichage
    df_metrics = pd.DataFrame([metrics_base, metrics_hybrid])
    
    print("\n🎯 MÉTRIQUES GLOBALES :")
    print(df_metrics[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    print("\n📈 MATRICE DE CONFUSION :")
    print(df_metrics[['Model', 'TP', 'TN', 'FP', 'FN']].to_string(index=False))
    
    # Analyse d'impact
    delta_fn = metrics_base['FN'] - metrics_hybrid['FN']
    delta_fp = metrics_hybrid['FP'] - metrics_base['FP']
    delta_f1 = metrics_hybrid['F1-Score'] - metrics_base['F1-Score']
    delta_recall = metrics_hybrid['Recall'] - metrics_base['Recall']
    
    print("\n" + "="*80)
    print("💡 ANALYSE D'IMPACT (Ce qui compte vraiment)")
    print("="*80)
    
    print(f"\n✅ VRAIS CANCERS SAUVÉS (Réduction FN) : {delta_fn:+d} patients")
    print(f"   Avant (DenseNet seul) : {metrics_base['FN']} cancers manqués")
    print(f"   Après (DenseNet + VAE): {metrics_hybrid['FN']} cancers manqués")
    
    print(f"\n⚠️  FAUSSES ALERTES AJOUTÉES (Augmentation FP) : {delta_fp:+d} patients")
    print(f"   Avant (DenseNet seul) : {metrics_base['FP']} fausses alertes")
    print(f"   Après (DenseNet + VAE): {metrics_hybrid['FP']} fausses alertes")
    
    print(f"\n📊 MÉTRIQUES CLÉS :")
    print(f"   Recall (Sensibilité) : {delta_recall:+.2f}% ({metrics_base['Recall']:.2f}% → {metrics_hybrid['Recall']:.2f}%)")
    print(f"   F1-Score : {delta_f1:+.2f}% ({metrics_base['F1-Score']:.2f}% → {metrics_hybrid['F1-Score']:.2f}%)")
    
    if delta_fn > 0:
        ratio = delta_fp / delta_fn if delta_fn > 0 else 0
        print(f"\n🚀 RATIO BÉNÉFICE/COÛT :")
        print(f"   Pour sauver 1 cancer, on accepte {ratio:.1f} fausses alertes supplémentaires")
        
        if delta_f1 > 0:
            print(f"\n🎉 VERDICT : VICTOIRE NETTE !")
            print(f"   ✓ Le système hybride sauve plus de cancers")
            print(f"   ✓ Le F1-Score augmente ({delta_f1:+.2f}%)")
            print(f"   ✓ Le système hybride est supérieur au DenseNet seul")
        elif delta_f1 > -2:
            print(f"\n✅ VERDICT : COMPROMIS CLINIQUE ACCEPTABLE")
            print(f"   ✓ Le système hybride sauve plus de cancers (+{delta_fn})")
            print(f"   ⚠️  Légère baisse du F1 ({delta_f1:.2f}%), mais acceptable cliniquement")
            print(f"   ✓ En médecine, sauver des cancers > optimiser le F1")
        else:
            print(f"\n⚠️  VERDICT : COMPROMIS À ÉVALUER")
            print(f"   ✓ Le système sauve {delta_fn} cancers supplémentaires")
            print(f"   ⚠️  Baisse significative du F1 ({delta_f1:.2f}%)")
            print(f"   → Considérer d'ajuster le seuil VAE (augmenter pour moins de FP)")
    else:
        print(f"\n❌ VERDICT : PAS D'AMÉLIORATION")
        print(f"   Le VAE n'ajoute que du bruit (FP) sans sauver de cancers")
        print(f"   → Action : Augmenter le seuil VAE (ex: {thresh_vae + 0.05:.3f})")
    
    # Sauvegarder les résultats
    os.makedirs(output_dir, exist_ok=True)
    
    df_metrics.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)
    print(f"\n✓ Métriques sauvegardées : {output_dir}/comparison_metrics.csv")
    
    # Visualisation des matrices de confusion côte à côte
    plot_comparison(metrics_base, metrics_hybrid, output_dir)


def plot_comparison(metrics_base, metrics_hybrid, output_dir):
    """Crée une visualisation comparative"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matrice DenseNet seul
    cm_base = np.array([[metrics_base['TN'], metrics_base['FP']], 
                        [metrics_base['FN'], metrics_base['TP']]])
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f"DenseNet SEUL\nF1={metrics_base['F1-Score']:.2f}%, Recall={metrics_base['Recall']:.2f}%",
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Matrice Hybride
    cm_hybrid = np.array([[metrics_hybrid['TN'], metrics_hybrid['FP']], 
                          [metrics_hybrid['FN'], metrics_hybrid['TP']]])
    
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    axes[1].set_title(f"HYBRIDE (DenseNet + VAE)\nF1={metrics_hybrid['F1-Score']:.2f}%, Recall={metrics_hybrid['Recall']:.2f}%",
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualisation sauvegardée : {output_dir}/comparison_confusion_matrices.png")


def main():
    parser = argparse.ArgumentParser(description='Évaluation Système Hybride DenseNet + VAE')
    parser.add_argument('--densenet_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='hybrid_evaluation_final')
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--densenet_threshold', type=float, default=0.3)
    parser.add_argument('--vae_threshold', type=float, default=0.136)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Charger les données
    print("\n📂 Chargement des données de test...")
    test_dataset = BinaryImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✓ {len(test_dataset)} images chargées")
    
    # Charger les modèles
    densenet = load_densenet(args.densenet_checkpoint, device)
    vae = load_vae(args.vae_checkpoint, args.latent_dim, device)
    
    # Évaluation
    df_results, y_pred_densenet, y_pred_hybrid = evaluate_hybrid_system(
        densenet, vae, test_loader, device,
        thresh_densenet=args.densenet_threshold,
        thresh_vae=args.vae_threshold
    )
    
    # Calculer les métriques
    y_true = df_results['label'].values
    metrics_base = calculate_metrics(y_true, y_pred_densenet, "DenseNet SEUL")
    metrics_hybrid = calculate_metrics(y_true, y_pred_hybrid, "HYBRIDE (DenseNet + VAE)")
    
    # Générer le rapport
    generate_comparison_report(metrics_base, metrics_hybrid, args.output_dir)
    
    # Sauvegarder les prédictions détaillées
    df_results.to_csv(os.path.join(args.output_dir, 'detailed_predictions.csv'), index=False)
    print(f"✓ Prédictions détaillées : {args.output_dir}/detailed_predictions.csv")
    
    print("\n" + "="*80)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*80)


if __name__ == '__main__':
    main()
