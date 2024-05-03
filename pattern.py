import cv2 as cv
import numpy as np
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
            patterns.append(pattern)
    return patterns

def find_best_matching_tile(image_path, pattern_folder):
    # Indlæs billedet
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return "Billedet kunne ikke indlæses."

    # Opdel billedet i tiles
    tiles = get_tiles(image)

    # Indlæs patterns
    patterns = load_patterns(pattern_folder)

    # Initialiser variabler til at finde det bedste match
    best_match_score = float('inf')
    best_match_position = (-1, -1)

    # Sammenlign hver tile med hvert pattern
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            for pattern in patterns:
                # Anvend matchTemplate og find det bedste match
                res = cv.matchTemplate(tile, pattern, cv.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                
                # Opdater det bedste match
                if min_val < best_match_score:
                    best_match_score = min_val
                    best_match_position = (x, y)

    return best_match_position

# Brug funktionen
image_path = 'sti_til_dit_billede'
pattern_folder = 'sti_til_dine_mønstre'
match_position = find_best_matching_tile(image_path, pattern_folder)
print(f"Bedste matchende tile er i position: {match_position}")