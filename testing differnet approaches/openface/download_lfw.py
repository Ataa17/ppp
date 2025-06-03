from sklearn.datasets import fetch_lfw_people
import os
import cv2

def download_and_save_lfw(output_dir='data/lfw_raw', min_faces=20):
    os.makedirs(output_dir, exist_ok=True)

    # Télécharger le dataset LFW (images couleur, funneled, toutes tailles, filtré)
    lfw = fetch_lfw_people(color=True, funneled=True, resize=1.0, min_faces_per_person=min_faces)
    print(f"Nombre de personnes : {len(lfw.target_names)}")

    for i, name in enumerate(lfw.target_names):
        person_dir = os.path.join(output_dir, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        idxs = [j for j, target in enumerate(lfw.target) if target == i]
        for count, idx in enumerate(idxs):
            # Image en float32 RGB [0.0, 1.0]
            img_float = lfw.images[idx]

            # Convertir en uint8 et en BGR pour OpenCV
            img_uint8 = (img_float * 255).astype('uint8')
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # Nom du fichier
            filename = os.path.join(person_dir, f"{name.replace(' ', '_')}_{count}.png")

            # Sauvegarde
            cv2.imwrite(filename, img_bgr)
            print(f"Image sauvegardée : {filename}")

if __name__ == "__main__":
    download_and_save_lfw()
