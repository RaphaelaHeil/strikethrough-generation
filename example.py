import cv2

import matplotlib.pyplot as plt

from strikethrough_generator import StrokeType, StrikeThroughGenerator

if __name__ == "__main__":
    stg = StrikeThroughGenerator(drawFromStrokeTypes=[StrokeType.ZIG_ZAG])
    original = cv2.imread('0004.png',cv2.CV_8UC1)
    output, strike_type = stg.generateStruckWord(original)

    output, _ = stg.generateStruckWord(original)

    plt.imshow(output, cmap="gray")
    plt.show()
