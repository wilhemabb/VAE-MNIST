```markdown
# Auto-encodeur Variationnel (VAE) pour la Représentation Latente et la Génération d'Images MNIST

## Description

Ce projet implémente un **Auto-encodeur Variationnel (VAE)** ainsi qu'un **Auto-encodeur de Wasserstein avec divergence MMD (WAE-MMD)** pour apprendre une représentation latente d'images de chiffres manuscrits de la base de données **MNIST**. Ces modèles capturent un espace latent compressé à partir duquel de nouveaux chiffres peuvent être générés en décodant des points échantillonnés de la loi a priori.

<p align="center">
  <img src="figures/vae_architecture.png" alt="Architecture du VAE" width="60%">
</p>

*Architecture du VAE : L'encodeur compresse les images d'entrée dans un espace latent, le décodeur reconstruit l'image à partir des variables latentes.*

## Résultats Clés
### Génération d'Images

Les modèles ont réussi à apprendre des représentations latentes structurées et exploitables pour le jeu de données MNIST, permettant des reconstructions et des générations d'images de haute qualité.

- **Reconstruction** : Le décodeur reconstruit fidèlement les images des chiffres à partir des variables latentes, prouvant que l'espace latent capture les informations structurelles essentielles des images d'entrée.
- **Génération** : En échantillonnant un bruit gaussien $\mathcal{N}(0, I)$ et en le soumettant au décodeur, le modèle est capable de synthétiser de nouvelles images réalistes de chiffres.

### Espace Latent
De plus, nous avons comparé l'effet de la **divergence de Kullback-Leibler (KL)** classique par rapport à l'**écart maximal moyen (Maximum Mean Discrepancy - MMD)** dans la fonction de coût :

- **MMD** : Génère un espace latent beaucoup plus structuré et sémantiquement continu. Les chiffres similaires sont regroupés de manière claire, et les relations géométriques locales sont mieux préservées.
- **KL** : Impose une contrainte stricte pour aligner chaque distribution a posteriori individuelle sur la loi normale standard, ce qui peut parfois compresser excessivement les régions latentes ou provoquer un phénomène d'inactivité de certaines dimensions.

## Aperçus Visuels

### Génération d'Images
<p align="center">
  <img src="figures/image_in.png" width="35%" style="margin-right: 5%;">
  <img src="figures/image_recon.png" width="35%">
</p>

*Image d'origine issue du jeu de test (à gauche) et sa reconstruction associée (à droite).*

<p align="center">
  <img src="figures/noise1.png" width="35%" style="margin-right: 5%;">
  <img src="figures/noise2.png" width="35%">
</p>

*Exemples de chiffres générés par le décodeur à partir d'un bruit blanc Gaussien.*

### Espace Latent (Visualisation 2D)
<p align="center">
  <img src="figures/kl.png" width="40%" style="margin-right: 5%;">
  <img src="figures/mmd.png" width="40%">
</p>

*Structure de l'espace latent projeté en 2D pour la perte KL (à gauche) et la perte MMD (à droite).*

## Fonctionnalités

- Implémentation complète de réseaux Encodeur et Décodeur
- Dimension de l'espace latent entièrement paramétrable (fixée à 200 par défaut)
- Utilisation de l'astuce de reparamétrisation (*reparameterization trick*) pour assurer la rétropropagation du gradient
- Entraînement sur MNIST avec une fonction de perte combinant la reconstruction (entropie croisée binaire) et la régularisation (divergence KL ou MMD)

## Jeu de Données

Le projet s'appuie sur le jeu de données officiel **MNIST** :

- 60 000 images d'entraînement
- 10 000 images de test
- Chiffres manuscrits en niveaux de gris de dimension 28x28 pixels
- Valeurs normalisées entre [0, 1]

## Structure du Projet

- `notebook.ipynb` : Notebook Jupyter contenant :
  - L'architecture complète du modèle (Encodeur, Décodeur)
  - Les calculs des fonctions de perte (ELBO classique et WAE-MMD)
  - La boucle d'entraînement et de validation
  - Les outils de visualisation de l'espace latent en 2D (via t-SNE) et la galerie de génération
- `paper.pdf` : Article scientifique de référence

## Méthodologie

- **Modèles** :
  - **Encodeur** : Un perceptron multicouche (MLP) projetant l'image $\rightarrow$ la moyenne ($\mu$) et le logarithme de la variance ($\log\sigma^2$)
  - **Astuce de reparamétrisation** pour l'échantillonnage differentiable : $z = \mu + \sigma \cdot \epsilon$ avec $\epsilon \sim \mathcal{N}(0, I)$
  - **Décodeur** : Un MLP projetant le vecteur latent $z$ pour reconstruire les pixels d'origine

- **Pertes** :
  - Perte de reconstruction : Entropie croisée binaire (BCE)
  - Terme de régularisation : Divergence KL classique ou calcul MMD non-paramétrique par noyau gaussien (RBF)

- **Entraînement** :
  - Optimiseur : Adam
  - Taux d'apprentissage : 0.001
  - Époques : 5
  - Taille des lots : 100

- **Bibliothèques utilisées** :
  - PyTorch pour l'implémentation et l'entraînement Deep Learning
  - Matplotlib pour la génération de graphiques
  - NumPy pour le traitement des données
  - Scikit-Learn (t-SNE) pour l'analyse et la projection de la variété latente

## Installation

Pour installer les bibliothèques requises, exécutez la commande suivante :

```bash
pip install torch torchvision numpy tqdm scikit-learn matplotlib
```
