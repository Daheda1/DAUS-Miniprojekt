import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def crown_detect(tile):
    return 1


def zoom_tile(tile, crop_percentage=1):
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
    height, width = tile.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = int(min(center_x, center_y) * circle_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    circle_removed_tile = cv.bitwise_and(tile, tile, mask=cv.bitwise_not(mask))
    return circle_removed_tile

def calculate_color_features(image):
    average_rgb = np.mean(image, axis=(0, 1))
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    average_hsv = np.mean(hsv_image, axis=(0, 1))
    flat_features = np.concatenate([average_rgb, average_hsv])

    return flat_features

def tile_data_get(tile):
    tile = zoom_tile(tile)
    tile = remove_circle_from_tile(tile)
    tile = calculate_color_features(tile)
    return tile

def get_terrain(tile):
    model = joblib.load('knn_model_model.joblib')
    tile_color_data = tile_data_get(tile)
    tile_color_data = tile_color_data.reshape(1, -1)
    terrain = model.predict(tile_color_data)

    return terrain

def get_tiles(image):
    image = cv.imread(image)

    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

def calculate_score(crownmatrix, terrainmatrix):
    total_score = 0
    visited = set()
    for y in range(5):
        for x in range(5):
            if (x, y) in visited:
                continue
            terrain_type = terrainmatrix[y][x]
            terrain_count, crown_count, _ = explore_terrain(terrainmatrix, crownmatrix, x, y, terrain_type, visited)
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

def matrix_create(imgpath):
    picturematrix = get_tiles(imgpath)

    crownmatrix = []
    terrainmatrix = []

    for x in range(5):
        crown_row = []
        terrain_row = []

        for y in range(5):
            crown_row.append(crown_detect(picturematrix[x][y]))
            terrain_row.append(get_terrain(picturematrix[x][y]))

        crownmatrix.append(crown_row)
        terrainmatrix.append(terrain_row)

    return crownmatrix, terrainmatrix, picturematrix


def show_score(crownmatrix, terrainmatrix, picturematrix):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ax = axes[i][j]
            ax.imshow(cv.cvtColor(picturematrix[i][j], cv.COLOR_BGR2RGB))
            label = f"{terrainmatrix[i][j]} {str(crownmatrix[i][j])}"
            ax.set_title(label, fontsize=8)  # Display the crown number
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    imgpath = "Data/KD train plader/1.jpg"
    crownmatrix, terrainmatrix, picturematrix = matrix_create(imgpath)
    score = calculate_score(crownmatrix, terrainmatrix)
    print(score)
    show_score(crownmatrix, terrainmatrix, picturematrix)


main()