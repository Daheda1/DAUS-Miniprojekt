import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from modeltrain import zoom_tile, remove_circle_from_tile, calculate_color_features

#Returnes alle the matrixes, one with alle images, one with terrain types and one with crown count
def matrix_create(imgpath):
    picturematrix = get_tiles(imgpath)

    crownmatrix = []
    terrainmatrix = []

    for x in range(5):
        crown_row = []
        terrain_row = []

        for y in range(5):
            crown_row.append(crown_detect(picturematrix[x][y], "Templates"))
            terrain_row.append(get_terrain(picturematrix[x][y]))

        crownmatrix.append(crown_row)
        terrainmatrix.append(terrain_row)

    return crownmatrix, terrainmatrix, picturematrix

#Splites one picture in to 25 smaller pictures
def get_tiles(image):
    image = cv.imread(image)

    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

#Find the amount of crowns in a picture
def crown_detect(img_bgr, template_path, threshold = 0.75, iou_thres = 0.2):
    boxes = []
    for template in os.listdir(template_path):
        actual_temp = os.path.join(r"Templates", template)
        current_template = cv.imread(actual_temp)
        assert current_template is not None, "file could not be read, check with os.path.exists()"

        # Rotate the template
        for rotation in range(4):
            temp = np.rot90(current_template, rotation)
            h, w, _ = temp.shape
            res = cv.matchTemplate(img_bgr, temp, cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            intersection_over_union(loc, h, w, boxes, img_bgr, iou_thres)
    return len(boxes)

#Makes sure any crown is only found once
def intersection_over_union(loc, h, w, boxes, img_bgr, iou_thres):
    for pts in zip(*loc[::-1]): 
        unions = []
        new_box = [pts[0], pts[1], pts[0] + w, pts[1] + h]

        if boxes == []:
            boxes.append(new_box)
            cv.rectangle(img_bgr, pts, (pts[0] + w, pts[1] + h), (0,0,255), 2)
        else:
            for box in boxes:
                # Determine the x and y coordinates of the intersection for the template rectangles
                xA = max(box[0], new_box[0])
                yA = max(box[1], new_box[1])
                xB = min(box[2], new_box[2])
                yB = min(box[3], new_box[3])

                # Find the area where the template rectangles intersect
                interArea = (xB - xA) * (yB - yA)

                # Find the area for both templates
                boxAArea = (box[2] - box[0]) * (box[3] - box[1])
                boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

                # Find the intersection over union and test that we don't divide by 0
                if interArea == 0 or float(boxAArea + boxBArea - interArea) == 0:
                    iou = 0
                else:
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                unions.append(iou)

            # Checks if the new box doesn't overlap with any of the other rectangles
            if all(i < iou_thres for i in unions):
                boxes.append(new_box)
                cv.rectangle(img_bgr, pts, (pts[0] + w, pts[1] + h), (0,0,255), 2)
    return boxes

#Predicts the terrain type
def get_terrain(tile):
    model = joblib.load('knn_model_model.joblib')

    tile_color_data = tile_data_get(tile)
    tile_color_data = tile_color_data.reshape(1, -1)

    terrain = model.predict(tile_color_data)

    return terrain

#Modifies tiles before getting data 
def tile_data_get(tile):
    tile = zoom_tile(tile)
    tile = remove_circle_from_tile(tile)
    tiledata = calculate_color_features(tile)
    return tiledata.reshape(1, -1)

#Does final score calculation
def calculate_score(crownmatrix, terrainmatrix):
    fullplate = 5
    total_score = 0
    visited = set()
    for y in range(5):
        for x in range(5):
            if (x, y) in visited:
                continue
            terrain_type = terrainmatrix[y][x]
            terrain_count, crown_count, _ = explore_terrain(terrainmatrix, crownmatrix, x, y, terrain_type, visited)
            total_score += terrain_count * crown_count

            if terrain_type == "None":
                fullplate = 0

    if terrainmatrix[2][2]  == "Home":
        total_score += 10

    return total_score+fullplate

#Debth first search for finding all connected terrains
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

#Shows all tiles with scores
def show_score(crownmatrix, terrainmatrix, picturematrix, score):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    plt.suptitle(f"Score: {score}", fontsize=16)

    for x in range(5):
        for y in range(5):
            ax = axes[x][y]
            ax.imshow(cv.cvtColor(picturematrix[x][y], cv.COLOR_BGR2RGB))
            label = f"{terrainmatrix[x][y]} {str(crownmatrix[x][y])}"
            ax.set_title(label, fontsize=8)  # Display the crown number
            ax.axis('off')
    plt.tight_layout()
    plt.show()

#Calculates diffrence between out systems score and the actual score
def calculate_dif_score(score, imgpath):
    actual_score = {"20.jpg": 67, "21.jpg": 55, "30.jpg": 63, "39.jpg": 62, "45.jpg": 43, "46.jpg": 48, "48.jpg": 57,
                    "49.jpg": 41, "50.jpg": 49, "55.jpg": 52, "63.jpg": 43, "65.jpg": 85, "67.jpg": 104, "70.jpg": 104}

    dif_score = abs(actual_score[imgpath] - score)

    return dif_score, actual_score[imgpath]

#The main
def main():
    imgpaths = r"Data/KD Test plader"
    for imgpath in os.listdir(imgpaths):
        if not imgpath.lower().endswith(('.jpg')):
            continue
        relative_imgpath = os.path.join(r"Data/KD Test plader", imgpath)
        crownmatrix, terrainmatrix, picturematrix = matrix_create(relative_imgpath)
        score = calculate_score(crownmatrix, terrainmatrix)
        dif_score, actual_score = calculate_dif_score(score, imgpath)
        print(f"On {imgpath} We predict that the final score is {score}, which is {dif_score} from the actual score of {actual_score}.")
        #show_score(crownmatrix, terrainmatrix, picturematrix, score)    

main()