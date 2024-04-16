import cv2
import numpy as np

# Indlæs billeder
img = cv2.imread('5.jpg', 0)  # Billedet, der skal søges i
template = cv2.imread('Krone.png', 0)  # Template-billede af en kongekrone
w, h = template.shape[::-1]

# Template matching
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  # Tærskelværdien for at bestemme en match
locations = np.where(res >= threshold)

# For hvert match, tegn en rektangel
for pt in zip(*locations[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
