import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from typing             import Tuple, List

def main() -> None:
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")

    image_path = r"/Users/dannihedegaarddahl/Documents/GitHub/daki_p0/King Domino dataset/22.jpg" #Billedsti programmet arbejder på

    if not os.path.isfile(image_path):                          #Tjekker at billedet eksistere
        print("Image not found")
        return 
    
    image = cv.imread(image_path)                               #Importere billede 
    tiles = get_tiles(image)                                    # Deler billedet i et 5x5 grid og opbevare dem i liste

    loaded_model = load_model("Models/tile_classifier_model.h5")#Loader modeller
    crown_model = load_model("Models/crown_detector_model.h5")

    terrains, crowns = generate_grid(tiles, loaded_model, crown_model)
    total_score = calculate_score(terrains, crowns)             #Beregner scoren for 
    
    print(f"Total Score: {total_score}")                        # Printer scoren for boardet i konsollen


#Deler billede i 5x5 Grid
def get_tiles(image) -> list:
    tiles = [] 
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles


def generate_grid(tiles: list, loaded_model, crown_model) -> Tuple[list, list]:
    terrains = []
    crowns = []

    fig, ax = plt.subplots(5, 5, figsize=(7, 7))

    for y, row in enumerate(tiles):
        terrains.append([])
        crowns.append([])
        for x, tile in enumerate(row):
            predicted_class, predicted_label, has_crown = get_terrain(tile, loaded_model, crown_model)
            terrains[-1].append(predicted_class)
            crowns[-1].append(has_crown)
            ax[y, x].imshow(cv.cvtColor(tile, cv.COLOR_BGR2RGB))
            ax[y, x].text(50, 20, f"{predicted_label}", fontsize=10, ha='center', va='center', color='white')
            ax[y, x].text(50, 50, f"({x},{y})", fontsize=10, ha='center', va='center', color='white')
            ax[y, x].axis('off')
    
    fig.tight_layout() 
    plt.show()
    return terrains, crowns

#Forbereder billede til tenserflow
def prepare_img(tile, width, height):
    img_array = cv.cvtColor(tile, cv.COLOR_BGR2RGB)         #Konvetere BGR til RGB
    img_array = cv.resize(img_array, (width, height))       #Sikre størrelse
    img_array = np.expand_dims(img_array, axis=0)           #tilføjer en ekstra dimension til billedet
    img_array = img_array.astype('float32') / 255.0         #Skallere farveværdierne fra [0,255] til [0,1]
    return img_array

# Finder ud af antallet af kroner i en tile
def count_crowns(tile, crown_model) -> int:
    img_array = prepare_img(tile, 100, 100)  # Forbereder hele tilet til den nye model
    predictions = crown_model.predict(img_array, verbose=0)  # Kalder den nye model

    predicted_class = np.argmax(predictions[0])

    labels = {0: 0, 1: 1, 2: 3, 3: 2}
    
    predicted_label = labels[predicted_class]

    return predicted_label  # Returnerer det forudsagte antal kroner

# Finder typen af tile og antal kroner for et tile
def get_terrain(tile, loaded_model, crown_model) -> Tuple[int, str, int]:
    img_array = prepare_img(tile, 100, 100)  # Behandler billede
    predictions = loaded_model.predict(img_array, verbose=0)  # Kalder model
    predicted_class = np.argmax(predictions[0])
    
    labels = {0: 'Field', 1: 'Forest', 2: 'Grassland', 
              3: 'Home', 4: 'Lake', 5: 'Mine', 
              6: 'None', 7: 'Swamp'}
    
    predicted_label = labels[predicted_class]  # Finder passende label baseret på prediction
    num_crowns = count_crowns(tile, crown_model)  # Tæller antal kroner
    
    if num_crowns > 0:  # Tilføjer antal kroner til label
        predicted_label += f" ({num_crowns} Crown{'s' if num_crowns > 1 else ''})"

    return predicted_class, predicted_label, num_crowns

def calculate_score(terrains, crowns):
    total_score = 0
    visited = set()
    for y in range(5):
        for x in range(5):
            if (x, y) in visited:
                continue
            terrain_type = terrains[y][x]
            terrain_count, crown_count, _ = explore_terrain(terrains, crowns, x, y, terrain_type, visited)
            total_score += terrain_count * crown_count
    return total_score

def explore_terrain(terrains, crowns, x, y, terrain_type, visited):
    if x < 0 or x >= 5 or y < 0 or y >= 5:
        return 0, 0, []
    if (x, y) in visited:
        return 0, 0, []
    if terrains[y][x] != terrain_type:
        return 0, 0, []
    visited.add((x, y))
    terrain_count = 1
    crown_count = int(crowns[y][x])
    tiles = [(x, y)]
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        sub_terrain_count, sub_crown_count, sub_tiles = explore_terrain(terrains, crowns, x + dx, y + dy, terrain_type, visited)
        terrain_count += sub_terrain_count
        crown_count += sub_crown_count
        tiles += sub_tiles
    return terrain_count, crown_count, tiles


if __name__ == "__main__":
    main()
