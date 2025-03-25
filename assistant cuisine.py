import re
import json
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import warnings
warnings.simplefilter("ignore", category=UserWarning)


# Charger le modèle d'embedding de spaCy
nlp = spacy.load("en_core_web_md")

# Charger la base de données des ingrédients
def clean_ingredient(ingredient):
    """ Nettoie et normalise le nom d’un ingrédient """
    ingredient = ingredient.lower()
    ingredient = re.sub(r"[^\w\s]", "", ingredient)
    return ingredient

with open("data/food_names.txt", "r", encoding="utf-8", errors="ignore") as fichier:
    contenu = fichier.read()
    ingredients_database = [clean_ingredient(line) for line in contenu.splitlines() if line.strip()]

# Générer les embeddings pour chaque ingrédient
ingredient_vectors = {ing: nlp(ing).vector for ing in ingredients_database}

# Fonction pour trouver l'ingrédient le plus similaire
def find_most_similar(target_ingredient):
    """ Trouve l'ingrédient le plus similaire dans la base de données """
    # Supprimer les parenthèses et normaliser
    target_ingredient = re.sub(r"\s*\([^)]*\)", "", target_ingredient).strip()

    # Vérification directe (singulier/pluriel)
    if target_ingredient in ingredient_vectors:
        return target_ingredient
    if target_ingredient + 's' in ingredient_vectors:
        return target_ingredient + 's'
    if target_ingredient + 'es' in ingredient_vectors:
        return target_ingredient + 'es'

    # Diviser en mots et vérifier chaque mot dans la base
    words = target_ingredient.split()
    for word in words:
        if word in ingredient_vectors:
            return word
        if word + 's' in ingredient_vectors:
            return word + 's'
        if word + 'es' in ingredient_vectors:
            return word + 'es'

    # Si aucun mot seul ne correspond, utiliser la similarité sémantique
    target_vec = nlp(target_ingredient).vector
    similarities = {
        ing: cosine_similarity([target_vec], [vec])[0][0] for ing, vec in ingredient_vectors.items()
    }

    # Retourne l'ingrédient avec la plus grande similarité
    return max(similarities, key=similarities.get, default=target_ingredient)

# Initialiser le modèle LLM
llm = ChatOpenAI(
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="deepseek/deepseek-chat:free"
)

def extract_info_from_user_input(user_input):
    prompt = (
        f"Extract the dish name and the number of people from the following input:\n"
        f"{user_input}\n\n"
        "Return the information in the following format:\n"
        "Dish Name: [Dish Name]\n"
        "Number of People: [Number of People]"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def get_recipe(dish_name, num_people):
    prompt = (
        f"I want the recipe for {dish_name} for {num_people} people. "
        "List only the main ingredients in grams in a table format without any labels above the table. "
        "Do not show any calculations. The table should have the following format:\n\n"
        "| [Ingredient 1] | [Amount in grams] |\n"
        "| [Ingredient 2] | [Amount in grams] |\n"
        "Ensure all measurements are in grams." \
        "And list the recipe steps."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# User input
user_input = input("Please enter your recipe request (for example, 'I want to cook apple pie for 4 people.'): ")

# Extraction des informations
dish_name, num_people = None, None
while not dish_name or not num_people:
    extracted_info = extract_info_from_user_input(user_input)
    
    lines = extracted_info.split("\n")
    for line in lines:
        if line.startswith("Dish Name:"):
            dish_name = line.replace("Dish Name:", "").strip()
        elif line.startswith("Number of People:"):
            num_people = line.replace("Number of People:", "").strip()
    
    if not dish_name or not num_people:
        print("Retrying extraction...")

# Récupération de la recette
recipe_text = get_recipe(dish_name, num_people)
print("| Ingredient | Quantity (g) |")
print(recipe_text)

# Extraction des ingrédients avec regex
pattern = r"\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"
matches = re.findall(pattern, recipe_text.lower())

# Nettoyage et formatage
ingredients_list = [{"ingrédient": re.sub(r"\s*\([^)]*\)", "", ing.strip()), "quantité (g)": int(qty)} for ing, qty in matches]

# Comparaison avec la base de données
for item in ingredients_list:
    ingredient_name = item["ingrédient"]
    most_similar = find_most_similar(ingredient_name)
    item["ingrédient similaire"] = most_similar


# --- Configurations ---
RECOMMENDATIONS_PATH = "data/recommandation_alimentation.csv"
NUTRITION_DB_PATH = "data/traited_data_composition_aliment.csv"  # À remplacer
RECIPE_SERVINGS = int(num_people)  # Pour le nombre de personnes spécifié

# --- Vérification du fichier CSV ---
with open(RECOMMENDATIONS_PATH, 'r') as file:
    for i, line in enumerate(file):
        if line.count(',') != 4:  # Remplacez 4 par le nombre attendu de colonnes - 1
            print(f"Ligne {i + 1} a un problème : {line}")

# --- Charger les données ---
nutrition_db = pd.read_csv(NUTRITION_DB_PATH)  # Votre DB avec les colonnes fournies
recommendations = pd.read_csv(RECOMMENDATIONS_PATH)

# --- Données de la recette ---
recipe_ingredients = ingredients_list

# --- Calculer les apports nutritionnels totaux ---
total_nutrients = pd.Series(dtype=float)

for ing in recipe_ingredients:
    ing_name = ing["ingrédient similaire"].lower()
    qty = ing["quantité (g)"] / 100  # Conversion en 100g
    
    # Trouver les données nutritionnelles
    ing_data = nutrition_db[nutrition_db['Food Name'].str.lower() == ing_name].iloc[0]
    
    # Ajouter au total (en multipliant par la quantité)
    for nutrient in nutrition_db.columns[2:]:  # Exclure Food Code et Food Name
        total_nutrients[nutrient] = total_nutrients.get(nutrient, 0) + ing_data[nutrient] * qty

# Convertir en DataFrame et ajuster pour 1 portion
total_per_serving = total_nutrients / RECIPE_SERVINGS
# --- Comparaison avec les recommandations ---
comparison = []
for _, row in recommendations.iterrows():
    nutrient = row['Nutrient']
    actual = total_per_serving.get(nutrient, 0)
    
    if pd.notna(row['Min Daily (g/mg)']) and pd.notna(row['Max Daily (g/mg)']):
        # Convertir les valeurs en numérique si possible
        min_rec = row['Min Daily (g/mg)']
        max_rec = row['Max Daily (g/mg)']
        
        # Convertir min_rec en nombre
        if isinstance(min_rec, str):
            if min_rec.startswith('<'):
                min_rec = float(min_rec[1:]) / 2  # Approx
            elif min_rec == '*':
                min_rec = 0
            else:
                try:
                    min_rec = float(min_rec)
                except ValueError:
                    min_rec = 0
        
        # Convertir max_rec en nombre
        if isinstance(max_rec, str):
            if max_rec == '*':
                max_rec = float(min_rec) * 1.5  # Approx
            else:
                try:
                    max_rec = float(max_rec)
                except ValueError:
                    max_rec = 0
        
        comparison.append({
            'Nutrient': nutrient,
            'Min Recommended': float(min_rec),
            'Max Recommended': float(max_rec),
            'Actual': actual
        })

comparison_df = pd.DataFrame(comparison)

# --- Liste des nutriments importants ---
important_nutrients = [
    "Fat (g)",
    "Energy (kcal) (kcal)",
    "Total sugars (g)",
    "Satd FA /100g FA (g)",
    "Cholesterol (mg)"
]

# --- Filtrer les recommandations et les données de la recette ---
comparison_filtered = [row for row in comparison if row['Nutrient'] in important_nutrients]
comparison_df_filtered = pd.DataFrame(comparison_filtered)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Configure all text to be bold by default
plt.rcParams['font.weight'] = 'bold'

# --- Create the dictionary for translations ---
nutrient_translations = {
    'Satd FA /100g FA (g)': 'Saturated Fatty Acids',
    'Total sugars (g)': 'Total Sugars',
    'Fat (g)': 'Fat',
    'Cholesterol (mg)': 'Cholesterol',
    'Energy (kcal) (kcal)': 'Calories'
}

# Apply translations
comparison_df_filtered['Nutrient_Display'] = comparison_df_filtered['Nutrient'].map(lambda x: nutrient_translations.get(x, x))

# Calculate percentages relative to the maximum recommendations
comparison_df_filtered['Percentage'] = (comparison_df_filtered['Actual'] / comparison_df_filtered['Max Recommended']) * 100

# Sort data by increasing percentage (so it displays in descending order from top to bottom)
comparison_df_filtered = comparison_df_filtered.sort_values(by='Percentage', ascending=True)

nutrients = comparison_df_filtered['Nutrient_Display']  # Use translated names
percentages = comparison_df_filtered['Percentage']

# Create the figure with more space at the bottom for horizontal text
plt.figure(figsize=(10, 6))

# Create horizontal bars
bars = plt.barh(nutrients, percentages, color='#1f77b4')

# Add bold annotations
for i, v in enumerate(percentages):
    plt.text(v + 1, i, f'{v:.1f}%', ha='left', va='center', fontsize=10, color='black', fontweight='bold')

# Add a vertical line at 30% to indicate the threshold of maximum recommendations
max_rec_value = 30
plt.axvline(x=max_rec_value, color='red', linestyle='--', linewidth=1)

# Add the maximum recommendation text at the bottom horizontally
max_x_val = max(percentages) * 1.05  # Use the largest value as a reference to position the text
plt.text(max_rec_value, -0.7, 'Maximum Recommendation (30%)', color='red', 
         ha='center', va='top', fontsize=9, fontweight='bold', rotation=0)

# Customization with bold text
plt.xlabel('', fontsize=12, labelpad=10)
plt.ylabel('', fontsize=12)
plt.title(f'Percentage of Macronutrients in the Dish "{dish_name}"\nCompared to Recommendations', 
          fontsize=14, pad=20, fontweight='bold')

# Make the y-axis labels bold
plt.yticks(fontweight='bold')

# Remove borders and X-axis
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])  # Remove X-axis ticks
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Adjust spacing and add more space at the bottom for horizontal text
plt.subplots_adjust(bottom=0.18)
plt.tight_layout()

# Save and display the graph
plt.savefig('nutrition_percentage_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

exceeding_nutrients = comparison_df_filtered[comparison_df_filtered['Percentage'] > 30]

# Définir une palette de couleurs
import matplotlib.pyplot as plt
color_map = plt.get_cmap('tab10', len(recipe_ingredients))

import matplotlib.pyplot as plt

# 🔹 AUGMENTER LA TAILLE DE LA FIGURE ET L'ESPACEMENT
plt.figure(figsize=(12, 6))  # Taille augmentée pour plus de lisibilité

# Ajuster l'espacement entre les barres
bar_height = 0.6  # Augmenter la hauteur des barres pour plus de clarté

# Pour chaque nutriment dépassant les recommandations
for idx, row in enumerate(exceeding_nutrients.itertuples()):
    nutrient = row.Nutrient
    nutrient_display = row.Nutrient_Display

    ingredient_contributions = []
    
    for i, ing in enumerate(recipe_ingredients):
        ing_name = ing["ingrédient similaire"].lower()
        qty = ing["quantité (g)"] / 100  

        ing_data = nutrition_db[nutrition_db['Food Name'].str.lower() == ing_name].iloc[0]
        nutrient_value = ing_data.get(nutrient, 0) * qty

        ingredient_contributions.append((ing["ingrédient"], nutrient_value, color_map(i)))

    # Trier les contributions par ordre décroissant
    ingredient_contributions.sort(key=lambda x: x[1], reverse=True)

    # Préparer les données pour le graphique
    ingredients = [x[0] for x in ingredient_contributions]
    contributions = [x[1] for x in ingredient_contributions]
    colors = [x[2] for x in ingredient_contributions]

    # Normalisation des contributions pour un affichage équilibré
    total_contribution = sum(contributions)
    normalized_contributions = [c / total_contribution for c in contributions] if total_contribution > 0 else []

    # Tracer les barres empilées
    left_offset = 0
    for i in range(len(ingredients)):
        percentage = (contributions[i] / total_contribution) * 100 if total_contribution > 0 else 0
        
        # Tracer la barre avec une hauteur ajustée
        plt.barh(nutrient_display, normalized_contributions[i], height=bar_height, 
                 color=colors[i], left=left_offset)
        
        # Ajouter du texte si la contribution est significative
        if percentage > 5:
            plt.text(left_offset + normalized_contributions[i] / 2, idx, 
                     f"{ingredients[i]}\n{percentage:.1f}%", 
                     ha='center', va='center', fontsize=10, fontweight='bold', color="white")

        left_offset += normalized_contributions[i]

# 🔹 AJOUTER UN TITRE PLUS LISIBLE
plt.title(f'Ingredient Contribution to Macronutrients Exceeding Recommendations for "{dish_name}"', 
          fontsize=14, pad=20, fontweight='bold')

# 🔹 ÉTIQUETTES ET ESTHÉTIQUE
plt.yticks(fontsize=11, fontweight='bold')  
plt.xticks([])  # Supprime les valeurs de l'axe X

# 🔹 SUPPRIMER LES BORDURES INUTILES
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

# 🔹 AJUSTER L'ESPACE AUTOUR DES BARRES
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)

# 🔹 SAUVEGARDER AVEC UN PADDING SUPPLÉMENTAIRE
plt.savefig('improved_nutrient_excess.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

# Afficher le graphique
plt.show()


# Génération des avertissements
warnings = []
for row in exceeding_nutrients.itertuples():
    nutrient = row.Nutrient
    nutrient_display = row.Nutrient_Display
    percentage = row.Percentage

    # Trouver l'ingrédient le plus responsable
    ingredient_contributions = []
    
    for ing in recipe_ingredients:
        ing_name = ing["ingrédient similaire"].lower()
        qty = ing["quantité (g)"] / 100  # Conversion en 100g

        ing_data = nutrition_db[nutrition_db['Food Name'].str.lower() == ing_name].iloc[0]
        nutrient_value = ing_data.get(nutrient, 0) * qty

        ingredient_contributions.append((ing["ingrédient"], nutrient_value))

    # Trouver l'ingrédient principal
    ingredient_contributions.sort(key=lambda x: x[1], reverse=True)
    main_ingredient, main_contribution = ingredient_contributions[0]
    
    # Calcul de la part de cet ingrédient dans l'excès
    total_contribution = sum(x[1] for x in ingredient_contributions)
    contribution_percentage = (main_contribution / total_contribution) * 100 if total_contribution > 0 else 0

    warnings.append(  
        f"Warning: The amount of {nutrient_display} exceeds the recommended limits "  
        f"({percentage:.1f}% of the recommended limit). The main responsible ingredient is "  
        f"{main_ingredient}, contributing {contribution_percentage:.1f}% to this excess."  
    )


print("\n")
# Display warnings
for warning in warnings:
    print(warning)

# Propose an alternative recipe if excess nutrients are detected
if warnings:
    user_choice = input("\nWould you like a healthier alternative recipe for this dish? (yes/no) : ").strip().lower()
    
    if user_choice == 'yes':
        # Create a prompt to generate a healthier recipe
        prompt = (
            f"The dish '{dish_name}' contains problematic nutrient excesses. "
            f"Here are the detected issues:\n"
            f"{chr(10).join(warnings)}\n\n"
            f"Please suggest an alternative recipe for a similar but healthier dish. "
            f"List the ingredients in grams in a table format and include the preparation steps. "
            f"Ensure the quantities are suitable for {num_people} people."
        )
        
        # Generate the alternative recipe
        alternative_recipe = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print("\nHealthier alternative recipe:\n", alternative_recipe)
    else:
        print("Alright, no alternative recipe will be generated.")
else:
    print("No nutrient excess detected. No alternative recipe needed.")