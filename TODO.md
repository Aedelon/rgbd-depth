# TODO - Actions Immédiates

## ✅ Déjà fait
- [x] Package standalone créé
- [x] Code migré et testé
- [x] CLI fonctionnels (cdm-infer, cdm-download)
- [x] Tests + CI/CD
- [x] Documentation complète

## 📋 À faire maintenant

### 1. Setup Git & GitHub (5 min)

```bash
cd /Users/aedelon/Workspace/camera-depth-models

# Éditer le script et remplacer TON-ORG par ton username GitHub
nano GIT_SETUP.sh  # Ligne 10

# Lancer le script automatique
./GIT_SETUP.sh
```

Le script va :
1. ✓ Init Git
2. ✓ Commit initial
3. ✓ Config remote SSH (git@github.com:...)
4. ✓ Demander de créer le repo sur https://github.com/new
5. ✓ Push le code
6. ✓ Créer le tag v1.0.2

### 2. Configurer GitHub (2 min)

Sur https://github.com/TON-USERNAME/camera-depth-models :

- [ ] **Settings → Actions → General**
  - Allow all actions ✓

- [ ] **Settings → Features**
  - Discussions ✓ (optionnel)

### 3. Créer la release (3 min)

1. Aller sur https://github.com/TON-USERNAME/camera-depth-models/releases/new
2. Choose tag: `v1.0.2`
3. Release title: `v1.0.2 - Initial Release`
4. Description: Copier depuis `MIGRATION_SUMMARY.md` section "Publication PyPI"

### 4. Mettre à jour le repo parent (2 min)

Éditer `/Users/aedelon/Workspace/manip-as-in-sim-suite/README.md` :

```markdown
## 📦 Components

### [Camera Depth Models (CDM)](https://github.com/TON-USERNAME/camera-depth-models)

**Standalone package** now available!

```bash
pip install camera-depth-models
```

Pre-trained models and documentation: [camera-depth-models repo](https://github.com/TON-USERNAME/camera-depth-models)

### WBCMimic

Enhanced MimicGen for mobile manipulators (this repository).
```

Puis commit :
```bash
cd /Users/aedelon/Workspace/manip-as-in-sim-suite
git add README.md
git commit -m "docs: Update README to reference standalone CDM package"
git push
```

## 🚀 Plus tard (optionnel)

### Publication PyPI

Quand tu es prêt à publier sur PyPI :

```bash
cd /Users/aedelon/Workspace/camera-depth-models

# Build
pip install build twine
python -m build

# Test sur TestPyPI d'abord
twine upload --repository testpypi dist/*

# Test l'installation depuis TestPyPI
pip install --index-url https://test.pypi.org/simple/ camera-depth-models

# Si tout va bien, publish sur PyPI production
twine upload dist/*
```

Après publication, les utilisateurs pourront faire :
```bash
pip install camera-depth-models
```

### Configurer PyPI publishing automatique

Dans le repo GitHub :
1. Settings → Secrets → Actions
2. New repository secret
3. Name: `PYPI_API_TOKEN`
4. Value: ton API token depuis https://pypi.org/manage/account/token/

Ensuite, chaque fois que tu crées une release, le workflow `.github/workflows/publish.yml` publiera automatiquement sur PyPI.

## 📝 Notes

- **URLs à remplacer** : Cherche `TON-ORG` ou `TON-USERNAME` dans tous les fichiers et remplace par ton vrai username GitHub
- **Fichiers concernés** :
  - `GIT_SETUP.sh` (ligne 10)
  - `pyproject.toml` (URLs)
  - `README.md` (badges et liens)
  - Ce fichier (TODO.md)

## ✅ Checklist de vérification finale

Avant de considérer le projet terminé :

- [ ] Git initialisé et code pushé sur GitHub
- [ ] GitHub Actions activés
- [ ] Release v1.0.2 créée
- [ ] Tests CI passent (vérifier les badges)
- [ ] Repo parent mis à jour
- [ ] Tous les `TON-ORG`/`TON-USERNAME` remplacés
- [ ] README badges fonctionnels

## 🎯 Résultat attendu

Une fois tout fait :
- ✅ Package standalone sur GitHub
- ✅ Installation : `pip install camera-depth-models`
- ✅ CLI : `cdm-infer`, `cdm-download`
- ✅ Tests automatiques
- ✅ Documentation exhaustive
- ✅ (Optionnel) Sur PyPI
