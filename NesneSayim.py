import cv2
import numpy as np
input_image = cv2.imread('nesne.jpeg')
gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
templates = [
    cv2.imread('karpuz.png', 0),
    cv2.imread('porta.png', 0),
    cv2.imread('elma.png', 0),
    cv2.imread('limon.png', 0),
    cv2.imread('ananas.png', 0),
    cv2.imread('kiraz.png', 0),
    cv2.imread('cilek.png', 0),
    cv2.imread('seftali.png', 0),
    cv2.imread('yesil.png', 0),
    cv2.imread('kivi.png', 0),
    cv2.imread('bisi.png', 0),
    cv2.imread('avakado.png', 0)
]
results = []
locations_list = []
for i, template in enumerate(templates):
    result = cv2.matchTemplate(gray_input, template, cv2.TM_CCOEFF_NORMED)
    results.append(result)
for i, result in enumerate(results):
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    locations = np.where(result >= max_val - 0.05)
    if len(locations[0]) > 0:
        loc = (max_loc[0], max_loc[1])
        cv2.rectangle(input_image, loc, (loc[0] + templates[i].shape[1], loc[1] + templates[i].shape[0]), (0, 255, 0), 2)
        cv2.putText(input_image, str(i + 1), (loc[0], loc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Tespit Edilen Nesneler', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
