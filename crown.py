import cv2
import os

def main():
    # Import tile
    image_path = r"1.jpg"
    output_folder = r"Data/KD train tiles"
    if len(os.listdir(output_folder)) == 1:
        make_tiles(image_path, output_folder)
    
    # Template matching
    tile = r"Data\KD train tiles\tile_1_3.jpg" # Data\KD train tiles\tile_0_0.jpg (0), Data\KD train tiles\tile_1_3.jpg (1), Data\KD train tiles\tile_3_3.jpg (2)
    template = r"Templates\1.jpg_1_2_0crowns.jpg" 
    template_matching(tile, template)
    
    # prøv sift

    # prøv noget andet

    #mere end 3, se bortfra
    # lav certainty, se bortfra

    # vote system måske? 

    # Print number of crowns

def make_tiles(image_path, output_folder):
    rows = 5
    columns = 5
    image = cv2.imread(image_path)

    # Antag en nær-ideel opdeling af pladen i et 5x5 grid
    tile_width = image.shape[1] // 5
    tile_height = image.shape[0] // 5

    for row in range(rows):
        for column in range(columns):
            left = column * tile_width
            upper = row * tile_height
            right = (column + 1) * tile_width
            lower = (row + 1) * tile_height

            tile = image[upper:lower, left:right]
            tile_path = f'{output_folder}/tile_{row}_{column}.jpg'
            cv2.imwrite(tile_path, tile)

def template_matching(tile, template):
    pass


main()