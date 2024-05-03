import cv2 as cv
import numpy as np
import random
import os

def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

def load_patterns(pattern_folder):
    patterns = []

    for filename in os.listdir(pattern_folder):
        path = os.path.join(pattern_folder, filename)
        pattern = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if pattern is not None:
            # Tilføj det oprindelige mønster og filnavnet
            patterns.append((pattern, filename))
            # Tilføj roterede versioner af mønsteret med filnavnet inkluderet
            rotated_pattern_90 = cv.rotate(pattern, cv.ROTATE_90_CLOCKWISE)
            patterns.append((rotated_pattern_90, filename + '_90'))
            rotated_pattern_180 = cv.rotate(pattern, cv.ROTATE_180)
            patterns.append((rotated_pattern_180, filename + '_180'))
            rotated_pattern_270 = cv.rotate(pattern, cv.ROTATE_90_COUNTERCLOCKWISE)
            patterns.append((rotated_pattern_270, filename + '_270'))

    return patterns


def find_best_matching_tile(tiles, patterns):
    best_match_score = float('inf')
    best_match_position = (-1, -1)
    best_pattern_name = None

    # Sammenlign hver tile med hvert pattern
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            for pattern, pattern_name in patterns:
                res = cv.matchTemplate(tile, pattern, cv.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                
                if min_val < best_match_score:
                    best_match_score = min_val
                    best_match_position = (x, y)
                    best_pattern_name = pattern_name

    return best_match_position, best_pattern_name


# Brug funktionen
#image_path = 'Data/KD Test plader/11.jpg'
#pattern_folder = 'Data/Home'
#match_position = find_best_matching_tile(image_path, pattern_folder)
#print(f"Bedste matchende tile er i position: {match_position}")


def draw_rectangle_on_match(image, best_match_position, tile_size=100):
    """Tegn en rød firkant rundt om det bedst matchende tile."""
    x, y = best_match_position
    top_left = (x * tile_size, y * tile_size)
    bottom_right = ((x + 1) * tile_size, (y + 1) * tile_size)
    marked_image = cv.rectangle(image.copy(), top_left, bottom_right, (0, 0, 255), 3)
    return marked_image

def process_images_in_folder(image_folder, pattern_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    patterns = load_patterns(pattern_folder)
    
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        image = cv.imread(file_path)
        if image is None:
            print(f"Kunne ikke indlæse billedet: {file_path}")
            continue

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tiles = get_tiles(gray_image)
        best_match_position, best_pattern_name = find_best_matching_tile(tiles, patterns)
        
        if best_match_position != (-1, -1):
            marked_image = draw_rectangle_on_match(image, best_match_position)
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, marked_image)
            print(f"Processed and saved {filename}; best matching pattern was {best_pattern_name}")


# Brug funktionen
image_folder = 'Data/KD Test plader'
pattern_folder = 'Data/Home'
output_folder = 'test'
process_images_in_folder(image_folder, pattern_folder, output_folder)
