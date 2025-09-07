# Fashion Trend Intelligence - Segmentation Vestimentaire

Un système d'analyse automatisé des tendances vestimentaires sur les réseaux sociaux pour ModeTrends.

## 📋 Contexte du Projet

ModeTrends, agence de conseil en marketing digital leader dans l'industrie de la mode, lance le projet "Fashion Trend Intelligence". Ce système vise à analyser automatiquement les tendances vestimentaires émergentes sur les réseaux sociaux, permettant aux marques clientes d'anticiper les modes avant qu'elles ne deviennent populaires.

### Objectifs Globaux du Système

Le système complet comprendra trois fonctionnalités principales :

1. **Segmentation vestimentaire** : Identification et isolation précise de chaque pièce vestimentaire dans une image
2. **Analyse stylistique** : Classification des pièces selon leur nature, couleur, texture et style
3. **Agrégation de tendances** : Compilation des données sur des milliers de publications pour identifier les tendances émergentes

## 🎯 Mission Actuelle

Ce projet se concentre sur la **première fonctionnalité : la segmentation vestimentaire**.

### Objectifs Spécifiques

-   Évaluer l'utilisation du modèle **SegFormer-clothes** via l'API Hugging Face
-   Tester la segmentation sur des images annotées
-   Effectuer un chiffrage des coûts d'utilisation de la solution API

## 🛠️ Technologies Utilisées

-   **Python** : Langage principal du projet
-   **Poetry** : Gestionnaire de dépendances et environnement virtuel
-   **JupyterLab** : Environnement de développement interactif
-   **Hugging Face API** : Service d'inférence pour le modèle SegFormer-clothes
-   **SegFormer-clothes** : Modèle pré-entraîné spécialisé dans la segmentation vestimentaire

## 📁 Structure du Projet

```
OC_P2_Fashion_Trend_Intelligence/
├── notebooks/
│   └── huggingface_api_cloth_seg.ipynb  # Notebook principal pour tester la segmentation vestimentaire
├── scripts/
│   └── test_hf_connection.py        # Script pour tester la connexion à l'API Hugging Face
├── utils/
│   └── connection_utils.py        # Utilitaires pour la connexion à l'API Hugging Face
│   └── image_utils.py              # Utilitaires pour le traitement d'images et masques
├── assets/                         ✅ (images de test, masques prédits, masques de vérité terrain)
├── pyproject.toml              # Configuration Poetry
├── poetry.lock                 # Verrouillage des dépendances
└── README.md
```

## 🚀 Installation et Configuration

### Prérequis

-   Python 3.12+
-   Poetry installé sur votre système

### Installation

1. Cloner le repository :

```bash
git clone <repository-url>
cd OC_P2_Fashion_Trend_Intelligence
```

2. Installer les dépendances avec Poetry :

```bash
poetry install
```

3. Activer l'environnement virtuel :

```bash
poetry shell
```

4. Lancer JupyterLab :

```bash
poetry run jupyter lab
```

Ouvrir le notebook `huggingface_api_cloth_seg.ipynb` pour commencer les tests.

### Configuration de l'API Hugging Face

1. Créer un compte sur [Hugging Face](https://huggingface.co/)
2. Obtenir votre clé API depuis les paramètres de votre compte
3. Configurer la clé API dans le notebook de test

## 📊 Tâches à Réaliser

### Tâche 1 : Test de Segmentation

-   [ ] Configurer l'accès à l'API Hugging Face
-   [ ] Tester le modèle SegFormer-clothes sur les images annotées
-   [ ] Évaluer la qualité de la segmentation
-   [ ] Documenter les résultats

### Tâche 2 : Analyse des Coûts

-   [ ] Calculer le coût pour 500 000 images sur 30 jours
-   [ ] Analyser les différentes options tarifaires
-   [ ] Proposer des recommandations d'optimisation

## 📈 Livrables Attendus

1. **Notebook d'analyse** : Tests du modèle SegFormer-clothes
2. **Rapport de coûts** : Évaluation financière de la solution API
3. **Présentation** : Synthèse des résultats pour l'équipe

## 🔍 Modèle Utilisé

**SegFormer-clothes** : Modèle de segmentation sémantique spécialement affiné pour identifier et segmenter les vêtements dans les images. Ce modèle est capable de :

-   Détecter différentes pièces vestimentaires
-   Créer des masques de segmentation précis
-   Traiter des images de qualité variable

## 📝 Notes Importantes

-   Ce projet constitue la première étape d'un système plus large d'analyse de tendances
-   Les résultats serviront de base pour les fonctionnalités suivantes
-   L'accent est mis sur la simplicité et la clarté technique pour la présentation

## 🧼 Bonnes Pratiques Notebooks (nbstripout)

Pour garder l'historique Git propre et éviter des diffs verbeux sur les sorties des notebooks, le dépôt utilise `nbstripout`.

### Installation locale (une seule fois)

```bash
poetry run nbstripout --install
```

Cela configure un filtre Git (défini dans `.gitattributes`) qui supprime les sorties (`outputs`, `execution_count`) lors des commits.

### Pourquoi ?

-   Diff plus lisibles (seulement le code change)
-   Moins de conflits de merge
-   Réduction de la taille du repo

### Désactiver temporairement

```bash
poetry run nbstripout --uninstall
```

Puis réactiver avec la commande d'installation.

### Stripper manuellement un notebook précis

```bash
poetry run nbstripout chemin/vers/notebook.ipynb
```

Voir `CONTRIBUTING.md` pour plus de détails.

## 👥 Équipe

-   **Développeur IA** - Me 🧑🏼‍🎓

## 📄 Licence

[À définir selon les politiques de ModeTrends]
