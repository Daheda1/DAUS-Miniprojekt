import cv2 as cv
import numpy as np
import os
import csv

def main():
    tiles_folder = "Data/KD train tiles/blandet"
    csv_document = "Data/blandet.csv"

    images = load_images_from_folder(tiles_folder)
    with open(csv_document, "w", newline='') as file:
        writer = csv.writer(file)
        # Skriv header til CSV-filen
        writer.writerow(["Filename", "Hue", "Saturation", "Value"])
        for filename, tile in images:
            hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
            hue, saturation, value = np.median(hsv_tile, axis=(0,1))
            print(filename, hue, saturation, value)
            write_to_csv(filename, hue, saturation, value, writer)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None and img.any():
            images.append((filename, img))
    return images

def write_to_csv(filename, hue, saturation, value, writer):
    writer.writerow([filename, hue, saturation, value])

main()
