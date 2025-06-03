import os
import shutil

# Obtenir le chemin absolu du dossier racine du projet (face_recog)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Définir les chemins source et destination
src_root = os.path.join(project_root, 'data', 'lfw_raw')
dest_flat = os.path.join(project_root, 'data', 'lfw_flat')

# Créer le dossier plat s’il n’existe pas
os.makedirs(dest_flat, exist_ok=True)

# Copier les images en les renommant avec le nom de la personne
for person in os.listdir(src_root):
    person_dir = os.path.join(src_root, person)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            src_img = os.path.join(person_dir, img_name)
            dest_img = os.path.join(dest_flat, f"{person}_{img_name}")
            shutil.copyfile(src_img, dest_img)

print("✅ Images copiées dans un dossier à plat avec nom d'identité.")
