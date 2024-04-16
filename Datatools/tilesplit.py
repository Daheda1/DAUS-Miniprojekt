import cv2
import numpy as np

# Indlæs billedet
image_path = 'Data/KD test plader/22.jpg' # Erstat med stien til dit billede
image = cv2.imread(image_path)

# Konverter til gråskala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Anvend Canny edge detector
edges = cv2.Canny(gray, threshold1=70, threshold2=130)

# Find konturer
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Antag en nær-ideel opdeling af pladen i et 5x5 grid
tile_width = image.shape[1] // 5
tile_height = image.shape[0] // 5

tiles = []
for cnt in contours:
    # Beregn omsluttende rektangel for hver kontur
    x, y, w, h = cv2.boundingRect(cnt)
    # Udsnit hvert tile baseret på det omsluttende rektangel
    # Bemærk: Du kan tilføje yderligere logik her for at sikre, at du kun udsnitter gyldige tiles
    if w > 30 and h > 30: # Juster disse værdier baseret på dine specifikke behov
        tile = gray[y:y+h, x:x+w]
        tiles.append(tile)

for i, tile in enumerate(tiles):
    cv2.imwrite(f'tile_{i}.jpg', tile) # Gemmer hvert tile som et billede
