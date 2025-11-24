#!/bin/bash
# Script de setup Git avec SSH pour camera-depth-models

set -e

echo "=========================================="
echo "Git Setup - Camera Depth Models"
echo "=========================================="
echo ""

# Configuration
GITHUB_ORG="Aedelon"
REPO_NAME="camera-depth-models"

echo "Configuration:"
echo "  Organisation/User: $GITHUB_ORG"
echo "  Repository: $REPO_NAME"
echo ""

# Vérifier si on est dans le bon répertoire
if [ ! -f "pyproject.toml" ] || [ ! -d "rgbddepth" ]; then
    echo "❌ Erreur: Exécute ce script depuis le répertoire camera-depth-models"
    exit 1
fi

# Vérifier que Git n'est pas déjà initialisé
if [ -d ".git" ]; then
    echo "⚠️  Git déjà initialisé dans ce répertoire"
    read -p "Continuer quand même ? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    # Initialiser Git
    echo "1. Initialisation Git..."
    git init
    echo "✓ Git initialisé"
fi

# Configurer .gitignore si besoin
if [ ! -f ".gitignore" ]; then
    echo "❌ .gitignore manquant !"
    exit 1
fi

# Ajouter tous les fichiers
echo ""
echo "2. Ajout des fichiers..."
git add .
echo "✓ Fichiers ajoutés"

# Premier commit
echo ""
echo "3. Premier commit..."
git commit -m "Initial commit: Camera Depth Models v1.0.2

- Standalone package extracted from manip-as-in-sim-suite
- CLI tools: cdm-infer, cdm-download
- Comprehensive tests with CI/CD
- Full documentation and guides
- Production-ready for PyPI
"
echo "✓ Commit créé"

# Renommer branche en main
echo ""
echo "4. Configuration de la branche main..."
git branch -M main
echo "✓ Branche renommée en main"

# Ajouter remote SSH
echo ""
echo "5. Configuration remote SSH..."
SSH_URL="git@github.com:${GITHUB_ORG}/${REPO_NAME}.git"
echo "   URL: $SSH_URL"

# Vérifier si remote existe déjà
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' existe déjà"
    echo "   Remote actuel: $(git remote get-url origin)"
    read -p "Remplacer par SSH ? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
        git remote add origin "$SSH_URL"
        echo "✓ Remote mis à jour avec SSH"
    fi
else
    git remote add origin "$SSH_URL"
    echo "✓ Remote ajouté"
fi

# Vérifier la clé SSH
echo ""
echo "6. Vérification de la connexion SSH GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ Authentification SSH réussie"
else
    echo "⚠️  Impossible de se connecter à GitHub via SSH"
    echo ""
    echo "Pour configurer SSH sur GitHub :"
    echo "  1. Générer une clé SSH (si nécessaire):"
    echo "     ssh-keygen -t ed25519 -C \"ton-email@example.com\""
    echo "  2. Ajouter la clé à ssh-agent:"
    echo "     eval \"\$(ssh-agent -s)\""
    echo "     ssh-add ~/.ssh/id_ed25519"
    echo "  3. Copier la clé publique:"
    echo "     cat ~/.ssh/id_ed25519.pub | pbcopy"
    echo "  4. L'ajouter sur GitHub:"
    echo "     https://github.com/settings/keys"
    echo ""
    read -p "Continuer quand même ? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Instructions pour créer le repo sur GitHub
echo ""
echo "=========================================="
echo "⚠️  AVANT DE PUSHER"
echo "=========================================="
echo ""
echo "1. Crée le repository sur GitHub :"
echo "   → https://github.com/new"
echo ""
echo "   Paramètres :"
echo "   • Owner: $GITHUB_ORG"
echo "   • Repository name: $REPO_NAME"
echo "   • Description: Camera Depth Models for accurate metric depth estimation from RGB-D sensors"
echo "   • Public ✓"
echo "   • ❌ NE PAS initialiser avec README (déjà présent)"
echo "   • ❌ NE PAS ajouter .gitignore (déjà présent)"
echo "   • ❌ NE PAS ajouter license (déjà présent)"
echo ""

read -p "Repository créé sur GitHub ? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Crée le repo puis relance : git push -u origin main"
    exit 0
fi

# Push
echo ""
echo "7. Push vers GitHub..."
git push -u origin main
echo "✓ Code pushé sur GitHub !"

# Créer le tag
echo ""
echo "8. Création du tag v1.0.2..."
git tag -a v1.0.2 -m "Release v1.0.2: Initial standalone release

Camera Depth Models v1.0.2 - First standalone release

Features:
- Metric depth estimation from RGB-D sensors
- Pre-trained models for RealSense, ZED 2i, Kinect
- Automatic device-specific optimizations (CUDA/MPS/CPU)
- CLI tools: cdm-infer, cdm-download
- Comprehensive tests and CI/CD
"
git push origin v1.0.2
echo "✓ Tag v1.0.2 créé et pushé"

echo ""
echo "=========================================="
echo "✅ Setup complet !"
echo "=========================================="
echo ""
echo "Repository : https://github.com/${GITHUB_ORG}/${REPO_NAME}"
echo ""
echo "Prochaines étapes :"
echo "  1. Activer GitHub Actions :"
echo "     → Settings → Actions → General → Allow all actions"
echo "  2. Activer Discussions (optionnel) :"
echo "     → Settings → Features → Discussions ✓"
echo "  3. Créer la release v1.0.2 :"
echo "     → https://github.com/${GITHUB_ORG}/${REPO_NAME}/releases/new"
echo "     → Choose tag: v1.0.2"
echo "     → Release title: v1.0.2 - Initial Release"
echo "     → Description: Voir MIGRATION_SUMMARY.md"
echo ""
echo "Pour publier sur PyPI plus tard :"
echo "  1. python -m build"
echo "  2. twine upload dist/*"
echo ""
