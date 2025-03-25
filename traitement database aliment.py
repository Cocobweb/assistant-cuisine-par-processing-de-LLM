import pandas as pd
import chardet

# Chemin vers votre fichier CSV
file_path = 'data/data_composition_aliment.csv'

# Détecter l'encodage du fichier
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

# Afficher l'encodage détecté
print(f"Encodage détecté : {result['encoding']}")

# Lire le fichier CSV avec l'encodage détecté
df = pd.read_csv(file_path, encoding=result['encoding'])

# Remplacer les cases vides (NaN) par 0
df.fillna(0, inplace=True)

# Remplacer les valeurs spécifiques comme 'N', 'Tr', etc., par 0
df.replace({'N': 0, '': 0}, inplace=True)

# Afficher les premières lignes pour vérifier
print(df.head())

print(df.dtypes)

# Liste des colonnes numériques (à adapter selon votre dataset)
numeric_columns = [
    'Water (g)', 'Total nitrogen (g)', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)', 
    'Energy (kcal) (kcal)', 'Energy (kJ) (kJ)', 'Starch (g)', 'Oligosaccharide (g)', 
    'Total sugars (g)', 'Glucose (g)', 'Galactose (g)', 'Fructose (g)', 'Sucrose (g)', 
    'Maltose (g)', 'Lactose (g)', 'Alcohol (g)', 'NSP (g)', 'AOAC fibre (g)', 
    'Satd FA /100g FA (g)', 'Satd FA /100g fd (g)', 'n-6 poly /100g FA (g)', 
    'n-6 poly /100g food (g)', 'n-3 poly /100g FA (g)', 'n-3 poly /100g food (g)', 
    'cis-Mono FA /100g FA (g)', 'cis-Mono FA /100g Food (g)', 'Mono FA/ 100g FA (g)', 
    'Mono FA /100g food (g)', 'cis-Polyu FA /100g FA (g)', 'cis-Poly FA /100g Food (g)', 
    'Poly FA /100g FA (g)', 'Poly FA /100g food (g)', 'Sat FA excl Br /100g FA (g)', 
    'Sat FA excl Br /100g food (g)', 'Branched chain FA /100g FA (g)', 
    'Branched chain FA /100g food (g)', 'Trans FAs /100g FA (g)', 
    'Trans FAs /100g food (g)', 'Cholesterol (mg)'
]


# Convertir les colonnes numériques en flottants
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remplacer les NaN par 0 (si nécessaire)
df.fillna(0, inplace=True)

# Vérifier les types de données après conversion
print(df.dtypes)

# Extraire le mot clé avant la première virgule
df['food_key'] = df['Food Name'].apply(lambda x: x.split(',')[0].strip())

# Créer un dictionnaire pour stocker le premier Food Code rencontré pour chaque groupe
first_food_codes = {}
for idx, row in df.iterrows():
    key = row['food_key']
    if key not in first_food_codes:
        first_food_codes[key] = row['Food Code']

# Grouper par food_key et calculer les moyennes
grouped_data = df.groupby('food_key')[numeric_columns].mean()

# Ajouter la colonne Food Code au dataframe groupé
grouped_data['Food Code'] = grouped_data.index.map(first_food_codes)

# Réinitialiser l'index pour transformer food_key en colonne
grouped_df = grouped_data.reset_index()

# Renommer la colonne food_key en Food Name
grouped_df = grouped_df.rename(columns={'food_key': 'Food Name'})

# Réorganiser les colonnes pour avoir Food Code en premier
cols = ['Food Code', 'Food Name'] + numeric_columns
grouped_df = grouped_df[cols]


for col in numeric_columns:
    grouped_df[col] = grouped_df[col].round(4)  # Utilisez 4 pour 4 chiffres après la virgule
# Afficher le résultat
print(grouped_df.head())
# Chemin de sortie pour le CSV
output_path = 'data/traited_data_composition_aliment.csv'

# Exporter en CSV
grouped_df.to_csv(output_path, index=False)

print(f"Données exportées avec succès vers {output_path}")

# Chemin de sortie pour le fichier texte
output_txt_path = 'data/food_names.txt'

# Écrire la liste des éléments de 'Food Name' dans un fichier texte
with open(output_txt_path, 'w') as f:
    for food_name in grouped_df['Food Name'].unique():
        f.write(f"{food_name}\n")

print(f"Liste des éléments de 'Food Name' exportée avec succès vers {output_txt_path}")