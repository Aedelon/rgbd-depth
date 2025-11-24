# Guide de Setup - Camera Depth Models

Ce guide t'accompagne pour configurer le nouveau repo standalone `camera-depth-models`.

## Étape 1 : Migration du code depuis l'ancien repo

```bash
cd camera-depth-models

# Exécuter le script de migration
./migrate_from_old_repo.sh ../manip-as-in-sim-suite

# Vérifier que tout est copié
ls -la rgbddepth/
ls -la example_data/
ls -la docs/
```

## Étape 2 : Initialiser Git

```bash
# Créer le repo Git local
git init

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "Initial commit: Camera Depth Models v1.0.2

- Standalone package extracted from manip-as-in-sim-suite
- Added CLI tools (cdm-infer, cdm-download)
- Added comprehensive tests
- Added CI/CD with GitHub Actions
- Optimized for CUDA/MPS/CPU
"

# Créer le repo sur GitHub (depuis l'interface web)
# Puis le lier :
git remote add origin https://github.com/TON-ORG/camera-depth-models.git
git branch -M main
git push -u origin main
```

## Étape 3 : Configuration GitHub

### 3.1 Activer GitHub Actions

1. Aller sur le repo GitHub
2. **Settings** → **Actions** → **General**
3. Autoriser les workflows

### 3.2 Configurer PyPI publishing (optionnel pour plus tard)

1. Créer compte sur [PyPI](https://pypi.org)
2. Créer un API token
3. Dans le repo GitHub : **Settings** → **Secrets** → **Actions**
4. Ajouter secret `PYPI_API_TOKEN` avec le token

### 3.3 Activer Discussions

1. **Settings** → **Features**
2. Cocher "Discussions"

## Étape 4 : Test local

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Installer en mode dev
pip install -e .[dev]

# Vérifier l'import
python -c "from rgbddepth import RGBDDepth, OptimizationConfig; print('✓ OK')"

# Lancer les tests
pytest tests/ -v

# Tester le CLI
cdm-download --list
cdm-infer --help
```

## Étape 5 : Vérifier le packaging

```bash
# Installer build tools
pip install build twine

# Build le package
python -m build

# Vérifier le contenu
tar -tzf dist/camera_depth_models-1.0.2.tar.gz | head -20

# Vérifier la validité
twine check dist/*

# Test d'installation depuis le build
pip install dist/camera_depth_models-1.0.2-py3-none-any.whl
python -c "import rgbddepth; print(rgbddepth.__version__)"
```

## Étape 6 : Créer la première release

```bash
# Créer un tag
git tag -a v1.0.2 -m "Release v1.0.2: Initial standalone release"
git push origin v1.0.2

# Sur GitHub :
# 1. Aller dans Releases
# 2. Draft new release
# 3. Choisir le tag v1.0.2
# 4. Titre : "v1.0.2 - Initial Release"
# 5. Description :
```

```markdown
## Camera Depth Models v1.0.2 - Initial Standalone Release

First standalone release of Camera Depth Models, extracted from the manip-as-in-sim-suite repository.

### Features
- ✅ Metric depth estimation from RGB-D sensors
- ✅ Pre-trained models for RealSense, ZED 2i, Kinect
- ✅ Automatic device-specific optimizations (CUDA/MPS/CPU)
- ✅ CLI tools: `cdm-infer`, `cdm-download`
- ✅ Python API with `OptimizationConfig`
- ✅ Comprehensive tests and CI/CD

### Installation
```bash
pip install camera-depth-models
```

### Supported Cameras
- Intel RealSense: D405, D415, D435, D455, L515
- Stereolabs ZED 2i: Quality, Neural modes
- Microsoft Azure Kinect

### Documentation
- [README.md](https://github.com/TON-ORG/camera-depth-models#readme)
- [OPTIMIZATIONS.md](./docs/OPTIMIZATIONS.md)
- [CHEATSHEET.md](./docs/CHEATSHEET.md)

### Related
- Paper: [Manipulation as in Simulation](https://manipulation-as-in-simulation.github.io/)
- Full suite: [manip-as-in-sim-suite](https://github.com/TON-ORG/manip-as-in-sim-suite)
```

## Étape 7 : Publication sur PyPI (quand prêt)

```bash
# Build le package
python -m build

# Upload sur PyPI (production)
twine upload dist/*

# Ou sur TestPyPI d'abord (recommandé)
twine upload --repository testpypi dist/*

# Test depuis TestPyPI
pip install --index-url https://test.pypi.org/simple/ camera-depth-models
```

## Étape 8 : Mise à jour du repo parent

Mettre à jour le README de `manip-as-in-sim-suite` pour pointer vers le nouveau repo :

```markdown
## 📦 Components

### [Camera Depth Models (CDM)](https://github.com/TON-ORG/camera-depth-models)

**Standalone package for depth estimation** - Now available separately!

```bash
pip install camera-depth-models
```

See the [CDM repository](https://github.com/TON-ORG/camera-depth-models) for:
- Pre-trained models for RealSense, ZED 2i, Kinect
- Easy CLI tools
- Full documentation

### WBCMimic

Enhanced MimicGen for mobile manipulators (this repository).
[...]
```

## Checklist final

- [ ] Code migré et fonctionne
- [ ] Tests passent localement
- [ ] Git initialisé et pushé
- [ ] GitHub Actions activé
- [ ] README complet avec badges
- [ ] Licence Apache 2.0 incluse
- [ ] CONTRIBUTING.md présent
- [ ] Première release créée (v1.0.2)
- [ ] PyPI publishing configuré (optionnel)
- [ ] Repo parent mis à jour

## Maintenance continue

### Versioning

Suivre [Semantic Versioning](https://semver.org/) :
- **MAJOR** (2.0.0) : Breaking changes
- **MINOR** (1.1.0) : Nouvelles features, backward compatible
- **PATCH** (1.0.1) : Bug fixes

### Release process

1. Mettre à jour version dans `pyproject.toml`
2. Mettre à jour CHANGELOG (à créer)
3. Commit : `git commit -m "Bump version to X.Y.Z"`
4. Tag : `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push : `git push && git push --tags`
6. GitHub Release → déclenche publish PyPI automatique

## Troubleshooting

### Tests échouent
```bash
# Vérifier Python version
python --version  # Doit être >= 3.8

# Réinstaller
pip install -e .[dev] --force-reinstall

# Nettoyer cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### Import errors
```bash
# Vérifier PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"

# Réinstaller en mode editable
pip uninstall camera-depth-models
pip install -e .
```

### CLI non trouvé
```bash
# Vérifier installation
pip show camera-depth-models

# Vérifier scripts
pip show -f camera-depth-models | grep cdm-

# Réinstaller
pip install --force-reinstall -e .
```

## Support

- Issues : [GitHub Issues](https://github.com/TON-ORG/camera-depth-models/issues)
- Discussions : [GitHub Discussions](https://github.com/TON-ORG/camera-depth-models/discussions)
