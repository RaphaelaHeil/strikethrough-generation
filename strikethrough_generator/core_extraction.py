"""
Core extraction based on:

A. Papandreou, B. Gatos,
Slant estimation and core-region detection for handwritten Latin words,
Pattern Recognition Letters, Volume 35, 2014, Pages 16-22, ISSN 0167-8655,
https://doi.org/10.1016/j.patrec.2012.08.005.

Implemented by R.Heil, 2021
"""
import itertools
from typing import Tuple

import numpy as np


def __runCountForLine__(line: np.ndarray, inkValue: int = 255) -> int:
    groups = []
    for _, g in itertools.groupby(line, lambda x: x == inkValue):
        groups.append(list(g))
    count = len([x for x in groups if inkValue in x])
    return count


def __countRuns__(image: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(__runCountForLine__, axis=1, arr=image)


def __calculateThreshold__(lines, t: float = 0.15) -> float:
    return t / len(lines) * sum(lines)


def __findCoreRegion__(booleanHorizontalProfile: np.ndarray, horizontalBlackRunProfile: np.ndarray) -> Tuple[int, int]:
    total = []
    borders = []
    current = 0
    start = 0
    for i, x in enumerate(booleanHorizontalProfile):
        if current == 0:
            start = i
        if x == 1:
            current += horizontalBlackRunProfile[i]
        else:
            if current > 0:
                total.append(current)
                borders.append((start, i - 1))
                current = 0
                start = i + 1
    if current > 0:
        total.append(current)
        borders.append((start, len(booleanHorizontalProfile) - 1))

    return borders[np.argmax(total)]


def extractCoreRegion(image: np.ndarray, thresholdModifier: float = 0.15) -> Tuple[int, int]:
    horizontalProfile = np.sum(image, 1)
    counts = __countRuns__(image)

    horizontalBlackRunProfile = counts * counts * horizontalProfile

    threshold = __calculateThreshold__(horizontalBlackRunProfile, thresholdModifier)
    booleanHorizontalProfile = (horizontalBlackRunProfile > threshold) * 1

    return __findCoreRegion__(booleanHorizontalProfile, horizontalBlackRunProfile)
