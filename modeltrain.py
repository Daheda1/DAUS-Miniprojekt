import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



def load_images_from_folders(root_folder):
    images_dict = {}
    
    # Gennemgå alle mapper i den angivne rodmappe
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        if os.path.isdir(folder_path):
            # Liste til at gemme billeder fra den aktuelle mappe
            images_list = []
            
            # Gennemgå alle filer i mappen
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # Indlæs billedet hvis filen er et billede
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image = cv.imread(file_path)
                    if image is not None:
                        images_list.append(image)
            
            # Tilføj listen af billeder til ordbogen med mappenavnet som nøgle
            if images_list:
                images_dict[folder_name] = images_list
    
    return images_dict


# Brug funktionen
root_folder = 'Data/KD train tiles'
images_dict = load_images_from_folders(root_folder)


def apply_transformations(images_dict):
    transformed_images_dict = {}
    
    for folder_name, images_list in images_dict.items():
        transformed_images = []
        
        for image in images_list:
            try:
                # Anvend zoom_tile funktionen
                zoomed_image = zoom_tile(image, crop_percentage=5)  # eksempel: 10% cropping

                # Anvend remove_circle_from_tile funktionen
                final_image = remove_circle_from_tile(zoomed_image, circle_size=0.5)  # eksempel: 30% cirkelstørrelse
                
                transformed_images.append(final_image)
            except Exception as e:
                print(f"En fejl opstod med billedbehandling: {e}")
        
        if transformed_images:
            transformed_images_dict[folder_name] = transformed_images
    
    return transformed_images_dict

def zoom_tile(tile, crop_percentage=1):
    if crop_percentage < 0 or crop_percentage > 100:
        raise ValueError("crop_percentage must be between 0 and 100")
    
    height, width = tile.shape[:2]
    crop_size = crop_percentage / 100.0
    start_x = int(width * crop_size / 2)
    start_y = int(height * crop_size / 2)
    end_x = width - start_x
    end_y = height - start_y
    cropped_tile = tile[start_y:end_y, start_x:end_x]
    zoomed_tile = cv.resize(cropped_tile, (width, height))
    return zoomed_tile

def remove_circle_from_tile(tile, circle_size=0.5):
    if circle_size <= 0 or circle_size > 1:
        raise ValueError("circle_size must be between 0 and 1")

    height, width = tile.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = int(min(center_x, center_y) * circle_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    circle_removed_tile = cv.bitwise_and(tile, tile, mask=cv.bitwise_not(mask))
    return circle_removed_tile

# Brug funktionen
transformed_dict = apply_transformations(images_dict)

def calculate_color_features(images_dict):
    features_dict = {}
    
    for folder_name, images_list in images_dict.items():
        folder_features = []
        
        for image in images_list:
            # Beregn gennemsnitlige og middel RGB værdier (som er det samme i dette tilfælde)
            average_bgr = np.mean(image, axis=(0, 1))
            
            # Konverter billede fra RGB til HSV
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            average_hsv = np.mean(hsv_image, axis=(0, 1))
            
            # Flad features ud til en enkelt liste
            flat_features = np.concatenate([average_bgr, average_hsv])
            
            # Tilføj den fladede feature-liste til mappens features
            folder_features.append(flat_features)
        
        if folder_features:
            features_dict[folder_name] = folder_features
    
    return features_dict

# Brug funktionen
color_features_dict = calculate_color_features(images_dict)

def train_knn(features_dict, n_neighbors=3):
    # Saml alle features og labels
    all_features = []
    all_labels = []

    for label, features_list in features_dict.items():
        for features in features_list:
            all_features.append(features)
            all_labels.append(label)
    
    # Konverter til numpy arrays for kompatibilitet med scikit-learn
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Opdel data i et trænings- og testsæt
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=40)

    # Initialiser kNN klassifikatoren
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Træn kNN-modellen
    knn.fit(X_train, y_train)

    return knn, X_test, y_test

def test_knn(knn, X_test, y_test):
    # Forudsige testdata
    y_pred = knn.predict(X_test)

    # Beregn nøjagtighed og andre metrikker
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Vis forvirringsmatrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Brug funktionerne
knn_model, X_test, y_test = train_knn(color_features_dict)
test_knn(knn_model, X_test, y_test)

joblib.dump(knn_model, 'knn_model_model.joblib')

