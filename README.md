# assistant-cuisine-par-processing-de-LLM
assistant cuisine par processing de LLM

## Module d'Analyse de Recettes et Nutrition

Ce module offre une solution complète pour l'analyse de recettes, la correspondance d'ingrédients et l'évaluation nutritionnelle en utilisant des technologies NLP et IA.

### Fonctionnalités

- **Correspondance d'ingrédients** :
  - Nettoie et normalise les noms d'ingrédients
  - Utilise les embeddings de spaCy pour une similarité sémantique
  - Gère les formes plurielles/singulières et les ingrédients composés

- **Traitement des recettes** :
  - Extrait le nom du plat et le nombre de portions depuis l'entrée utilisateur
  - Génère des recettes complètes avec ingrédients et étapes via API OpenAI
  - Structure les données de la recette

- **Analyse nutritionnelle** :
  - Calcule les valeurs nutritionnelles par portion
  - Compare aux recommandations alimentaires
  - Identifie les excès de nutriments
  - Visualise les données nutritionnelles avec matplotlib

- **Recommandations santé** :
  - Signale les ingrédients contribuant aux excès
  - Propose des alternatives plus saines si nécessaire

### Dépendances

- Python 3.x
- Paquets requis :
spacy
pandas
numpy
matplotlib
scikit-learn
langchain-openai
python-dotenv


### Fichiers de Données Requis

Placer ces fichiers dans le dossier `data/` :
- `food_names.txt` - Base de données d'ingrédients
- `traited_data_composition_aliment.csv` - Données nutritionnelles
- `recommandation_alimentation.csv` - Recommandations alimentaires

### Exemple d'Utilisation

```python
# L'utilisateur demande une recette
user_input = "Je veux faire un gratin dauphinois pour 4 personnes"

# Le système va :
# 1. Extraire le nom du plat et le nombre de portions
# 2. Générer la recette complète
# 3. Analyser le contenu nutritionnel
# 4. Fournir des visualisations
# 5. Proposer des alternatives plus saines si besoin
