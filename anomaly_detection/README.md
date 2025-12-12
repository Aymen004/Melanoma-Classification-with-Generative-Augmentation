# VAE-based Anomaly Detection Module

## 📋 Vue d'ensemble

Ce module implémente un système de détection d'anomalies basé sur un **Variational Autoencoder (VAE)** pour compléter le pipeline de classification des mélanomes.

### Pourquoi cette approche ?

L'approche VAE **renverse le problème de classification** :
- Au lieu d'apprendre à quoi ressemble un cancer, on apprend ce qu'est la **"normalité"** (lésions bénignes)
- Tout ce qui s'éloigne trop de cette normalité est signalé comme **anomalie**

### Avantages

1. **Indépendance vis-à-vis des données rares** : Pas besoin de nombreux exemples de mélanomes
2. **Filet de sécurité Out-of-Distribution (OOD)** : Détecte les cas jamais vus à l'entraînement
3. **Complémentaire au classificateur supervisé** : Réduit les faux négatifs critiques

---

## 🔗 Synergies avec le Pipeline Global

Ce module ne remplace pas le DenseNet, il le **renforce** via 3 synergies clés:

### ✅ Synergie 1: Triage Préliminaire (IMPLÉMENTÉ)

Le système fonctionne en deux temps:

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     Anomalie?
│   VAE Check     │────────► OUI ──► ⚠️  Priorité HAUTE pour dermatologue
└─────────┬───────┘                    (quelle que soit l'avis du DenseNet)
          │
          │ NON (Normal)
          ▼
┌─────────────────┐
│ DenseNet Clf    │────────► Classification standard
└─────────────────┘
```

**Code:**
```python
from anomaly_detection import HybridClassifier

hybrid = HybridClassifier(
    vae_model_path='vae_output/best_model.pth',
    classifier_model_path='models/densenet.pth',
    fusion_strategy='cascade'  # ← Triage préliminaire
)
```

### ✅ Synergie 2: XAI et Cartes d'Erreur (IMPLÉMENTÉ)

Le VAE offre une **explicabilité native** via les heatmaps d'anomalie:

```python
# Générer les heatmaps d'anomalie
fig, heatmaps = detector.generate_anomaly_heatmaps(
    images=test_images,
    colormap='hot'
)

# Superposer sur l'image originale (très intuitif pour les médecins)
overlay, score = detector.generate_overlay_heatmap(
    image=lesion_image,
    alpha=0.5
)
```

**Visualisation:**
```
┌──────────────┬──────────────┬─────────────────┐
│   Original   │ Reconstruction│ Heatmap Anomalie│
├──────────────┼──────────────┼─────────────────┤
│     🔵      │      🔵      │     [COOL]      │  ← Zone normale
│    🔴🔴     │     🔵🔵     │   [🔥 HOT]      │  ← Zone pathologique
│     🔵      │      🔵      │     [COOL]      │
└──────────────┴──────────────┴─────────────────┘
```

La heatmap montre **exactement** les zones que le VAE ne peut pas reconstruire!

### ✅ Synergie 3: Utilisation du DDPM (IMPLÉMENTÉ)

Si vous manquez de données bénignes variées, utilisez le **DDPM existant** pour générer plus de données d'entraînement saines:

```bash
# 1. Générer 1000 images bénignes avec DDPM
python anomaly_detection/ddpm_benign_augmentation.py \
    --ddpm_model_path generators/ddpm/checkpoints/best_model.pth \
    --num_samples 1000 \
    --quality_filter \
    --output_dir ./benign_augmented

# 2. Combiner avec les vraies données (70% réel, 30% synthétique)
python anomaly_detection/ddpm_benign_augmentation.py \
    --ddpm_model_path generators/ddpm/checkpoints/best_model.pth \
    --num_samples 500 \
    --real_data_dir ./data/benign_real \
    --synthetic_ratio 0.3 \
    --combined_output_dir ./benign_combined

# 3. Entraîner le VAE sur le dataset enrichi
python anomaly_detection/train_vae.py \
    --img_dir ./benign_combined \
    --epochs 100
```

**Avantage:** VAE plus robuste à la diversité normale de la peau (différents types de peau, éclairages, âges, etc.)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IMAGE D'ENTRÉE                            │
│                         (128×128×3)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                         ENCODEUR (CNN)                           │
│  Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128) → Conv2d(128→256) │
│                      + BatchNorm + LeakyReLU                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ESPACE LATENT                               │
│            μ (mean) ─────┬───── log(σ²) (log variance)          │
│                          │                                       │
│              z = μ + σ × ε  (Reparameterization Trick)          │
│                     dim = 256                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DÉCODEUR (CNN⁻¹)                          │
│  ConvT(256→128) → ConvT(128→64) → ConvT(64→32) → ConvT(32→3)    │
│                      + BatchNorm + LeakyReLU                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    IMAGE RECONSTRUITE                            │
│                         (128×128×3)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Fonction de Perte

$$\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot D_{KL}$$

### Perte de Reconstruction (MSE)
$$\mathcal{L}_{\text{reconstruction}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

Force le modèle à bien recréer les images saines.

### Divergence KL
$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

Régularise l'espace latent vers une distribution normale standard $\mathcal{N}(0, I)$.

---

## 🚀 Utilisation

### 1. Entraînement du VAE (uniquement sur images bénignes)

```bash
python train_vae.py \
    --img_dir ./data/isic2016/benign \
    --epochs 100 \
    --batch_size 32 \
    --latent_dim 256 \
    --beta 1.0 \
    --output_dir ./vae_output
```

### 2. Calibrage du Seuil

```python
from inference_vae import VAEAnomalyDetector

# Charger le modèle
detector = VAEAnomalyDetector(model_path='vae_output/checkpoints/best_model.pth')

# Calibrer sur un jeu de validation mixte
detector.calibrate(
    val_dataloader=val_loader,
    labels=val_labels,  # 0=bénin, 1=malin
    percentile=95.0,    # Seuil au 95ème percentile des bénins
    method='percentile'
)

# Visualiser la distribution des erreurs
detector.plot_error_distribution(
    errors_benign=benign_errors,
    errors_malignant=malignant_errors,
    save_path='error_distribution.png'
)
```

### 3. Inférence

```python
# Prédire pour un batch
predictions, anomaly_scores = detector.predict(images)

# Prédire pour une seule image
result = detector.predict_single(image_path='lesion.jpg')
print(f"Anomaly: {result['is_anomaly']}, Score: {result['anomaly_score']:.4f}")
```

### 4. Classificateur Hybride (VAE + DenseNet)

```python
from hybrid_classifier import HybridClassifier

# Créer le classificateur hybride
hybrid = HybridClassifier(
    vae_model_path='vae_output/checkpoints/best_model.pth',
    classifier_model_path='models/densenet_best.pth',
    fusion_strategy='weighted'  # ou 'voting', 'cascade', 'ensemble'
)

# Calibrer
hybrid.calibrate(val_loader, val_labels)

# Prédire
predictions, details = hybrid.predict(test_loader, return_details=True)

# Évaluer
metrics = hybrid.evaluate(test_loader, test_labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Improvement over classifier alone: {metrics['improvement_over_classifier']:.4f}")
```

---



1. Passer les images de validation dans le VAE
2. Calculer l'erreur de reconstruction (MSE) pour chaque image
3. Tracer la distribution des erreurs
4. Fixer le seuil entre les deux groupes (ex: 95ème percentile des bénins)

---

## 📁 Structure des Fichiers

```
anomaly_detection/
├── __init__.py              # Exports du module
├── VAE_model.py             # Architecture du VAE
├── train_vae.py             # Script d'entraînement
├── inference_vae.py         # Inférence et calibrage
├── hybrid_classifier.py     # Fusion VAE + DenseNet
└── README.md                # Cette documentation
```

---

## ⚙️ Configuration

### VAEConfig

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `image_size` | 128 | Taille des images en entrée |
| `latent_dim` | 256 | Dimension de l'espace latent |
| `hidden_dims` | [32, 64, 128, 256, 512] | Dimensions des couches cachées |
| `beta` | 1.0 | Coefficient KL (β-VAE) |
| `learning_rate` | 1e-4 | Taux d'apprentissage |
| `batch_size` | 32 | Taille des batchs |
| `dropout` | 0.2 | Taux de dropout |

---

## 📈 Métriques d'Évaluation

Le module calcule automatiquement :

- **Accuracy** : Précision globale
- **Sensitivity (Recall)** : Taux de vrais positifs (malins correctement détectés)
- **Specificity** : Taux de vrais négatifs (bénins correctement identifiés)
- **Precision** : Précision des prédictions positives
- **F1-Score** : Moyenne harmonique precision/recall
- **ROC-AUC** : Aire sous la courbe ROC
- **PR-AUC** : Aire sous la courbe Precision-Recall

---

## 🔬 Stratégies de Fusion

| Stratégie | Description | Usage recommandé |
|-----------|-------------|------------------|
| `voting` | Vote majoritaire (OR) | Maximiser le rappel |
| `weighted` | Moyenne pondérée | Équilibre precision/recall |
| `cascade` | VAE en premier filtre | Détection OOD prioritaire |
| `ensemble` | Combinaison avec boost | Performance optimale |

---

## 📚 Références

- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes
- Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
- An, J., & Cho, S. (2015). Variational Autoencoder based Anomaly Detection using Reconstruction Probability
