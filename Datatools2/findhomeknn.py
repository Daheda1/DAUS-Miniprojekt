import cv2 as cv
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Funktion til at beregne farveegenskaber fra en mappe
def calculate_color_features_from_folder(folder_path):
    features_dict = {}
    folder_features = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            continue

        # Beregn gennemsnitlige RGB og HSV værdier
        average_rgb = np.mean(image, axis=(0, 1))
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        average_hsv = np.mean(hsv_image, axis=(0, 1))
        
        # Flad features ud til en enkelt liste
        flat_features = np.concatenate([average_rgb, average_hsv])
        folder_features.append(flat_features)
    
    if folder_features:
        features_dict[os.path.basename(folder_path)] = folder_features

    return features_dict

# Funktion til at træne KNN model
def train_knn(features_dict, n_neighbors=3):
    X = []
    y = []

    for label, features in features_dict.items():
        X.extend(features)
        y.extend([label] * len(features))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    return knn, X_train, X_test, y_train, y_test

# Funktion til at forberede data og træne KNN
def prepare_and_train_knn(home_folder, not_home_folder, n_neighbors=3):
    home_features = calculate_color_features_from_folder(home_folder)
    not_home_features = calculate_color_features_from_folder(not_home_folder)
    
    # Combine features into a single dictionary with appropriate labels
    combined_features = {**home_features, **not_home_features}
    
    knn_model, X_train, X_test, y_train, y_test = train_knn(combined_features, n_neighbors=n_neighbors)
    
    return knn_model, X_train, X_test, y_train, y_test

# Funktion til at klassificere et nyt billede
def classify_image(tile, knn_model):
    if tile.ndim == 2 or tile.shape[2] == 1:  # Tjek om billedet er i gråtone
        tile = cv.cvtColor(tile, cv.COLOR_GRAY2BGR)
    average_rgb = np.mean(tile, axis=(0, 1))
    hsv_image = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    average_hsv = np.mean(hsv_image, axis=(0, 1))
    flat_features = np.concatenate([average_rgb, average_hsv])
    prediction = knn_model.predict([flat_features])[0]
    return prediction


def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles


def draw_rectangle_on_match(image, matches, tile_size=100):
    """Tegn en rød firkant rundt om matchende tiles."""
    marked_image = image.copy()
    for (x, y) in matches:
        top_left = (x * tile_size, y * tile_size)
        bottom_right = ((x + 1) * tile_size, (y + 1) * tile_size)
        cv.rectangle(marked_image, top_left, bottom_right, (0, 0, 255), 3)
    return marked_image

def process_images_in_folder(image_folder, output_folder, knn_model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        image = cv.imread(file_path)
        if image is None:
            print(f"Kunne ikke indlæse billedet: {file_path}")
            continue

        tiles = get_tiles(image)  # Brug den oprindelige farvebillede
        matches = []

        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                prediction = classify_image(tile, knn_model)
                if prediction == 'Home':  # Antag 'Home' som en klasse
                    matches.append((x, y))

        marked_image = draw_rectangle_on_match(image, matches)
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, marked_image)
        print(f"Processed and saved {filename}")


def classify_image(tile, knn_model, threshold=1):
    # Konverter til BGR hvis tile er i gråtone
    if tile.ndim == 2 or (tile.ndim == 3 and tile.shape[2] == 1):
        tile = cv.cvtColor(tile, cv.COLOR_GRAY2BGR)
    
    # Beregn gennemsnitlige RGB og HSV værdier
    average_rgb = np.mean(tile, axis=(0, 1))
    hsv_image = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    average_hsv = np.mean(hsv_image, axis=(0, 1))
    
    # Forbered features til klassifikation
    flat_features = np.concatenate([average_rgb, average_hsv])
    
    # Forudsige sandsynligheder for klasserne
    probabilities = knn_model.predict_proba([flat_features])[0]
    
    # Find klassen med den højeste sandsynlighed
    class_label = knn_model.classes_[np.argmax(probabilities)]
    
    # Tjek om den højeste sandsynlighed overstiger tærsklen
    if max(probabilities) <= threshold :
        return 'Home'
    else:
        return 'Not Home'



image_folder = 'Data/KD Test plader'
output_folder = 'test'
home_folder = 'Data/Home'
not_home_folder = 'Data/Not Home'
knn_model, X_train, X_test, y_train, y_test = prepare_and_train_knn(home_folder, not_home_folder)
process_images_in_folder(image_folder, output_folder, knn_model)
