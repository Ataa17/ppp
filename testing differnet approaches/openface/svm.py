# main_svm.py
import joblib

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from detect_common_columns import get_common_columns

# === 1. Dossier principal
embeddings_root = r"C:\Users\Admin\Documents\face_recog\embeddings"

# === 2. Détecter les colonnes communes à tous les fichiers
common_cols = get_common_columns(embeddings_root)
if 'label' in common_cols:
    common_cols.remove('label')  # on va l'ajouter manuellement plus tard

# === 3. Initialisation
all_data = []

# === 4. Parcours des sous-dossiers
for person_dirname in os.listdir(embeddings_root):
    person_dir = os.path.join(embeddings_root, person_dirname)

    if os.path.isdir(person_dir):
        csv_filename = person_dirname + ".csv"
        csv_path = os.path.join(person_dir, csv_filename)

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Vérifier que toutes les colonnes communes sont présentes
                if all(col in df.columns for col in common_cols):
                    subset = df[common_cols].copy()
                    subset["label"] = person_dirname
                    all_data.append(subset)
                else:
                    print(f"⚠️ Colonnes manquantes dans {csv_filename}, ignoré.")
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {csv_filename} : {e}")

# === 5. Fusion
if not all_data:
    raise ValueError("❌ Aucune donnée exploitable trouvée.")

df_all = pd.concat(all_data, ignore_index=True)
print(f"✅ {len(df_all)} images chargées")
print(f"✅ {df_all['label'].nunique()} personnes détectées")

# === 6. Préparation des données
X = df_all.drop(columns=["label"])
y = df_all["label"]

# === 7. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 8. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === 9. Entraînement SVM
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# === 10. Évaluation
y_pred = clf.predict(X_test)

#sauvegarder le modèle et le StandardScaler
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Modèle et scaler sauvegardés.")


print("\n✅ Rapport de classification :")
print(classification_report(y_test, y_pred))

print("✅ Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
