# Contributing to Camera Depth Models

Merci de ton intérêt pour contribuer à Camera Depth Models !

## Code de conduite

Sois respectueux et professionnel dans toutes les interactions.

## Comment contribuer

### Signaler un bug

1. Vérifie que le bug n'a pas déjà été signalé dans [Issues](https://github.com/manipulation-as-in-simulation/camera-depth-models/issues)
2. Crée une nouvelle issue avec :
   - Description claire du problème
   - Étapes pour reproduire
   - Environnement (OS, Python version, GPU/CPU)
   - Messages d'erreur complets

### Proposer une fonctionnalité

1. Ouvre une issue pour discuter de la fonctionnalité
2. Attends les retours avant de commencer le développement
3. Référence l'issue dans ta pull request

### Soumettre du code

1. **Fork** le repo
2. **Créer une branche** : `git checkout -b feature/ma-fonctionnalite`
3. **Coder** en suivant les conventions du projet
4. **Ajouter des tests** pour toute nouvelle fonctionnalité
5. **Vérifier** :
   ```bash
   # Tests
   pytest tests/ -v

   # Formatage
   black rgbddepth/ tests/
   isort rgbddepth/ tests/

   # Linting
   ruff check rgbddepth/ tests/
   ```
6. **Commit** : `git commit -m "Add: description claire"`
7. **Push** : `git push origin feature/ma-fonctionnalite`
8. **Pull Request** vers `main`

### Conventions de code

- **Formatage** : Black (line length 100)
- **Imports** : isort avec profil black
- **Docstrings** : Google style
- **Type hints** : Fortement encouragés pour les API publiques

### Structure des commits

```
Type: Description courte (50 chars max)

Description détaillée si nécessaire.

Fixes #123
```

Types : `Add`, `Fix`, `Update`, `Refactor`, `Docs`, `Test`

## Développement local

```bash
# Clone
git clone https://github.com/TON-USERNAME/camera-depth-models.git
cd camera-depth-models

# Install en mode dev
pip install -e .[dev,all]

# Run tests
pytest tests/ -v

# Format
black . && isort .
```

## Tests

Tous les PRs doivent passer les tests CI :
- Tests unitaires sur Python 3.8-3.11
- Tests sur Ubuntu, macOS, Windows
- Vérifications de formatage (black, isort, ruff)

Ajoute des tests pour :
- Nouvelles fonctionnalités
- Corrections de bugs (test de non-régression)

## Documentation

- Met à jour le README si l'API publique change
- Ajoute des docstrings pour les nouvelles fonctions/classes
- Met à jour OPTIMIZATIONS.md pour les changements de performance

## Questions

Ouvre une [Discussion](https://github.com/manipulation-as-in-simulation/camera-depth-models/discussions) pour les questions générales.

Merci ! 🙏
