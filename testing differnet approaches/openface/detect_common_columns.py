# detect_common_columns.py

import os
import pandas as pd

def get_common_columns(embeddings_root):
    all_columns = []
    valid_files = 0

    for person_dirname in os.listdir(embeddings_root):
        person_dir = os.path.join(embeddings_root, person_dirname)
        if os.path.isdir(person_dir):
            csv_filename = person_dirname + ".csv"
            csv_path = os.path.join(person_dir, csv_filename)

            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    all_columns.append(set(df.columns))
                    valid_files += 1
                except Exception as e:
                    print(f"❌ Erreur de lecture : {csv_filename} – {e}")
            else:
                print(f"⚠️ Fichier CSV non trouvé dans {person_dir}")

    if not all_columns:
        raise ValueError("❌ Aucun fichier CSV valide trouvé dans les sous-dossiers.")

    common_cols = set.intersection(*all_columns)
    print(f"✅ {valid_files} fichiers valides analysés.")
    print(f"✅ {len(common_cols)} colonnes communes détectées.")
    return sorted(common_cols)

# Pour test manuel
if __name__ == "__main__":
    folder = r"C:\Users\Admin\Documents\face_recog\embeddings"
    commons = get_common_columns(folder)
    print("Colonnes communes :", commons)
