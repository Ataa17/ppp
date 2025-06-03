import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

def local_binary_pattern(image, P=8, R=1):
    lbp = np.zeros_like(image, dtype=np.uint8)
    for i in range(R, image.shape[0]-R):
        for j in range(R, image.shape[1]-R):
            center = image[i, j]
            binary_string = ''
            for p in range(P):
                theta = 2 * np.pi * p / P
                x = i + R * np.sin(theta)
                y = j + R * np.cos(theta)
                x1, y1 = int(np.floor(x)), int(np.floor(y))
                x2, y2 = int(np.ceil(x)), int(np.ceil(y))
                tx, ty = x - x1, y - y1
                if x1 >= 0 and x2 < image.shape[0] and y1 >= 0 and y2 < image.shape[1]:
                    val = (1 - tx) * (1 - ty) * image[x1, y1] + tx * (1 - ty) * image[x2, y1] + \
                          (1 - tx) * ty * image[x1, y2] + tx * ty * image[x2, y2]
                    binary_string += '1' if val > center else '0'
            lbp[i, j] = int(binary_string, 2)
    return lbp

def lbp_histogram(img, radius=1, n_points=8):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, n_points, radius)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def load_dataset_from_sklearn():
    print("Fetching dataset from sklearn...")
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
    X_raw = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names

    print("Images shape:", X_raw.shape)
    print("Number of classes:", len(target_names))
    print("Computing LBP features...")

    X_features = []
    for idx, img in enumerate(X_raw):
        hist = lbp_histogram(img)
        X_features.append(hist)

    return np.array(X_features), y, target_names

if __name__ == "__main__":
    X, y, target_names = load_dataset_from_sklearn()
    print(f"Loaded {len(X)} samples with {len(target_names)} distinct people")

    if len(X) == 0:
        print("No data to train.")
        exit()

    print("Training SVM model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(classification_report(y_test, y_pred, zero_division=0))


    joblib.dump({'model': clf, 'target_names': target_names}, "lfw_face_recognition_model.pkl")
    print("Model saved as lfw_face_recognition_model.pkl")

    lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.5)
    plt.imshow(lfw_people.images[0], cmap='gray')
    plt.title(lfw_people.target_names[lfw_people.target[0]])
    plt.axis('off')
    plt.show()
