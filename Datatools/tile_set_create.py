import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program
def main():
    # foldere til billeder defineres
    output_folder = "Data/KD train tiles"
    input_folder = "Data/KD train plader"

    # vi kigger igennem hver fil i input folderen
    for filename in os.listdir(input_folder):
        # da billederne er navngivet som 1.jpg, 2.jpg osv. 
        # bruger vi filename counteren til at definere billedet vi vil have
        image_path = input_folder +"/" +filename
        # vi udskriver stien til billedet som man kan kigge på hvis der skulle opstå fejl
        print(image_path)
        # her tjekker vi om billedet findes i mappen og skriver "Image not found" hvis det ikke eksisterer
        if not os.path.isfile(image_path):
            print("Image not found")
            return
        
        # her bruger vi openCV til at åbne billedet
        image = cv.imread(image_path)
        tiles = get_tiles(image)
        print(len(tiles))
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                zoomed_tile = zoom_tile(tile)
                ready_tile = remove_circle_from_tile(zoomed_tile)
                save_tile(ready_tile, output_folder, filename, x, y)

# her definerer vi en funktion 'save_tile' med inputparametrene tile, outputfolder, image_name, x og y
def save_tile(tile, output_folder, image_name, x, y):
    # her laver vi en mappe 'blandet' hvis den ikke findes
    if not os.path.exists(os.path.join(output_folder, "blandet")):
        os.makedirs(os.path.join(output_folder, "blandet"))

    # her definerer vi navnet på det tile der skal gemmes som f.eks. 1_3_2.png ved brug af en f-streng
    tile_filename = f"{image_name}_{x}_{y}.png"

    # her definerer bi tile_path som er stedet hvor vi vil gemme vores tile
    tile_path = os.path.join(output_folder, "blandet", tile_filename)
    # her gemmer vi vores tile som tile_filename i folderen 'blandet'
    cv.imwrite(tile_path, tile)

    # her skriver vi til konsollen at vi har gemt vores til i 'blandet' folderen
    print(f"Saved Tile as {tile_filename} in 'blandet' folder")

def get_tiles(image):
    # laver en tom liste 
    tiles = []
    # kører et for loop hvor elementer vil blive tilføjet til listen, hvor y repræsenterer en række af billedet
    for y in range(5):
        tiles.append([])
    # kører et nested loop, hvor billedet bliver delt op i en tavel med 25 kvadrater af 100,100 px og tilføjer til listen tiles
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

def zoom_tile(tile, crop_percentage=0.10):
    height, width = tile.shape[:2]
    
    # Beregner hvor meget der skal beskæres fra hver side
    crop_height = int(height * crop_percentage)
    crop_width = int(width * crop_percentage)
    
    # Opdaterer udsnittets start- og slutpunkter for at fjerne de yderste 5%
    cropped_tile = tile[crop_height:height-crop_height, crop_width:width-crop_width]
    
    return cropped_tile

def remove_circle_from_tile(tile):
    # Konverterer til RGBA hvis billedet ikke allerede er det
    if tile.shape[2] == 3:
        tile = cv.cvtColor(tile, cv.COLOR_BGR2BGRA)
    
    height, width = tile.shape[:2]
    # Finder diameteren for cirklen som 5% af den mindste dimension af billedet
    diameter = int(0.70 * min(width, height))
    
    # Beregner centrum for cirklen
    center = (width // 2, height // 2)
    
    # Tegner cirklen i alfakanalen
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, center, diameter // 2, (255), -1)
    
    # Fjerner cirklen fra alfakanalen, så den bliver gennemsigtig
    tile[mask == 255, 3] = 0

    return tile

# her kører vi main() funktionen
if __name__ == "__main__":
    main()