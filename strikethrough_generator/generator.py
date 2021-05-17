import itertools
import logging
from enum import Enum, auto
from math import floor
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from imgaug.augmenters import ElasticTransformation
from scipy import interpolate

from .backgroundremoval import backgroundRemoval
from .core_extraction import extractCoreRegion


class StrokeType(Enum):
    SINGLE_LINE = 0
    DOUBLE_LINE = 1
    DIAGONAL = 2
    CROSS = 3
    WAVE = 4
    ZIG_ZAG = 5
    SCRATCH = 6


class OptionsKeys(Enum):
    PAD = auto()
    STROKE_WIDTH = auto()
    CORE_REGION = auto()
    MIN_SIZE = auto()


def setDefaultOptions(options: Dict[OptionsKeys, Any]):
    if OptionsKeys.PAD not in options:
        options[OptionsKeys.PAD] = 10
    if OptionsKeys.MIN_SIZE not in options:
        options[OptionsKeys.MIN_SIZE] = (60, 60)


class StrikeThroughGenerator:
    TRANSFORM = ElasticTransformation(alpha=(0, 20.0), sigma=(4.0, 6.0))

    def __init__(self, options: Dict[OptionsKeys, Any] = None, drawFromStrokeTypes: List[StrokeType] = None,
                 seed: Any = None):
        np.random.seed(seed)
        if drawFromStrokeTypes is None:
            self.drawFromStrokeTypes = list(StrokeType)
        else:
            self.drawFromStrokeTypes = drawFromStrokeTypes
        if options:
            self.options = options
        else:
            self.options = {}
        setDefaultOptions(self.options)

    def generateStruckWord(self, original: np.ndarray, strokeType: StrokeType = None) -> np.ndarray:
        """
        Generates a strikethrough stroke and applies it to the given word. If :param:`strokType` is not given, a random one
        will be drawn from :class:`StrokeType`

        Parameters
        ----------
        original : np.ndarray
            word image to be struck through

        strokeType : StrokeType
            stroke type to be applied to the image. If `None` a random one will be drawn from :class:`StrokeType`

        Returns
        -------
        np.ndarray
            the struck-through word image

        """
        if not strokeType:
            strokeType = np.random.choice(self.drawFromStrokeTypes)

        height, width = original.shape
        if OptionsKeys.MIN_SIZE in self.options:
            minSize = self.options[OptionsKeys.MIN_SIZE]
            assert height >= minSize[0]
            assert width >= minSize[1]
        original = 255 - self.__removeBackground(original).astype(np.uint8)
        binaryImage = self.__binarise(original)

        if OptionsKeys.STROKE_WIDTH in self.options:
            strokeWidth = self.options[OptionsKeys.STROKE_WIDTH]
        else:
            strokeWidth = self.__getStrokeWidth(binaryImage)
        if OptionsKeys.CORE_REGION in self.options:
            coreRegion = self.options[OptionsKeys.CORE_REGION]
        else:
            coreRegion = extractCoreRegion(binaryImage)

        imageDT = self.__dt(binaryImage)
        averageInkValue = self.__calculateAverageInkValue(original)

        if strokeType == StrokeType.SINGLE_LINE:
            lineCoordinates = self.__singleLine(height, width, coreRegion)
            lineProfile = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates)], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineDT = self.__lineDT(lineProfile, self.options[OptionsKeys.PAD])
            line = self.__generateStrokeFromDT(original, imageDT, lineDT)
            strikeThrough = self.__deformLine(line)
        elif strokeType == StrokeType.DOUBLE_LINE:
            lineCoordinates = self.__doubleLine(height, width, coreRegion)
            lineProfile = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates[0:2])], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineProfile = cv2.polylines(lineProfile, [np.array(lineCoordinates[2:4])], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineDT = self.__lineDT(lineProfile, self.options[OptionsKeys.PAD])
            line = self.__generateStrokeFromDT(original, imageDT, lineDT)
            strikeThrough = self.__deformLine(line)
        elif strokeType == StrokeType.DIAGONAL:
            lineCoordinates = self.__diagonal(height, width, coreRegion=coreRegion)
            lineProfile = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates[0:2])], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineProfile = cv2.polylines(lineProfile, [np.array(lineCoordinates[2:4])], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineDT = self.__lineDT(lineProfile, self.options[OptionsKeys.PAD])
            line = self.__generateStrokeFromDT(original, imageDT, lineDT)
            strikeThrough = self.__deformLine(line)
        elif strokeType == StrokeType.CROSS:
            lineCoordinates = self.__cross(height, width, coreRegion=coreRegion)
            lineProfile1 = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates[0:2])], 0, (255, 255, 255),
                                         thickness=int(strokeWidth))
            lineProfile2 = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates[2:4])], 0, (255, 255, 255),
                                         thickness=int(strokeWidth))
            lineProfile1DT = self.__lineDT(lineProfile1, self.options[OptionsKeys.PAD])
            lineProfile2DT = self.__lineDT(lineProfile2, self.options[OptionsKeys.PAD])
            line1 = self.__generateStrokeFromDT(original, imageDT, lineProfile1DT)
            line2 = self.__generateStrokeFromDT(original, imageDT, lineProfile2DT)
            line = np.zeros(original.shape, dtype=np.int32)
            line += line1
            line += line2
            line = np.clip(line, 0, 255)
            strikeThrough = self.__deformLine(line.astype(np.uint8))
        elif strokeType == StrokeType.WAVE:
            lineCoordinates = self.__wave(height, width, coreRegion=coreRegion)
            lineProfile = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates)], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineDT = self.__lineDT(lineProfile, self.options[OptionsKeys.PAD])
            line = self.__generateStrokeFromDT(original, imageDT, lineDT)
            strikeThrough = self.__deformLine(line)
        elif strokeType == StrokeType.ZIG_ZAG:
            lineCoordinates = self.__zigZag(height, width, core_region=coreRegion)
            lineProfile = cv2.polylines(np.zeros_like(imageDT), [np.array(lineCoordinates)], 0, (255, 255, 255),
                                        thickness=int(strokeWidth))
            lineDT = self.__lineDT(lineProfile, self.options[OptionsKeys.PAD])
            line = self.__generateStrokeFromDT(original, imageDT, lineDT)
            strikeThrough = self.__deformLine(line)
        elif strokeType == StrokeType.SCRATCH:
            lineSegments = self.__scratch(height, width, coreRegion=coreRegion)
            lineImages = []
            previousX = lineSegments[0][0]
            previousY = lineSegments[0][1]
            for i in range(1, len(lineSegments)):
                currentX = lineSegments[i][0]
                currentY = lineSegments[i][1]
                canvas = np.zeros(original.shape)
                cv2.line(canvas, (previousX, previousY), (currentX, currentY), (255, 255, 255),
                         thickness=int(strokeWidth))
                canvas_dt = self.__lineDT(canvas)
                generated_stroke = self.__generateStrokeFromDT(original, imageDT, canvas_dt)
                lineImages.append(generated_stroke)
                previousX = currentX
                previousY = currentY
            total = np.sum(np.array(lineImages), axis=0)
            total[total > 255] = 255
            strikeThrough = self.__deformLine(total.astype(np.uint8))
        else:
            raise NotImplementedError('No implementation found for {}'.format(strokeType))
        return (255 - self.__superimpose(original, strikeThrough, averageInkValue)), strokeType

    def __removeBackground(self, original: np.ndarray) -> np.ndarray:
        bgrResult = backgroundRemoval(original, 1, 5, 0.0, False)
        mask = (bgrResult != 255)
        backgroundMasked = np.ones_like(bgrResult) * 255
        backgroundMasked[mask] = original[mask]
        return backgroundMasked

    def __calculateAverageInkValue(self, original: np.ndarray) -> int:
        tmp = original.copy()
        tmp = tmp > 0.0
        return floor(np.sum(original) / (np.sum(tmp) + 0.000001))

    def __superimpose(self, original: np.ndarray, strikeThrough: np.ndarray, averageInkValue: int) -> np.ndarray:
        superimposedImage = original.copy().astype(np.float32)

        maxInkValue = superimposedImage.max()
        factor = averageInkValue / maxInkValue

        line = strikeThrough.copy()
        tmp = superimposedImage + line
        indices = np.logical_and(superimposedImage > averageInkValue, line > averageInkValue)
        tmp[indices] = (superimposedImage[indices] + line[indices] * factor)

        tmp2 = tmp.copy()
        tmp2 = cv2.GaussianBlur(tmp2, (3, 3), 0.8)
        tmp[line > 0] = tmp2[line > 0]
        tmp = np.clip(tmp, 0, 255)

        return tmp.astype(np.uint8)

    def __generateStrokeFromDT(self, original: np.ndarray, imageDT: np.ndarray, lineDT: np.ndarray) -> np.ndarray:
        syntheticLine = np.zeros_like(original)
        distances = np.unique(lineDT)[1:]
        drawFrom = original
        for val in distances:
            colourPool = drawFrom[imageDT == val]
            if len(colourPool) == 0:
                colourPool = drawFrom[imageDT > 0]
            indices = np.argwhere(lineDT == val)
            for index in indices:
                syntheticLine[index[0], index[1]] = np.random.choice(colourPool)
        return syntheticLine

    def __binarise(self, image: np.ndarray) -> np.ndarray:
        _, binarised = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarised

    def __lineDT(self, lineImage: np.ndarray, pad: int = 10) -> np.ndarray:
        height, width = lineImage.shape
        paddedLine = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
        paddedLine[pad:pad + height, pad:pad + width] = lineImage[:, :]
        lineDT = self.__dt(paddedLine)
        lineDT = lineDT[pad:pad + height, pad:pad + width]
        return lineDT

    def __dt(self, image: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform(image, cv2.DIST_L2, 5)

    def __deformLine(self, line: np.ndarray) -> np.ndarray:
        deformedLine = self.TRANSFORM.augment_image(line)
        return deformedLine

    def __runLength(self, line: np.ndarray, inkValue: int = 255) -> List[int]:
        groups = []
        for _, g in itertools.groupby(line, lambda x: x == inkValue):
            groups.append(list(g))
        length = [len(x) for x in groups if inkValue in x]
        return length

    def __getStrokeWidth(self, binaryImage: np.ndarray, medianModifier: float = 0.75) -> float:
        lengths = []
        height, width = binaryImage.shape
        for i in range(height):
            line = binaryImage[i]
            lengths.extend(self.__runLength(line))
        for j in range(width):
            line = binaryImage[:, j]
            lengths.extend(self.__runLength(line))
        imageMedian = np.median(lengths)
        return imageMedian * medianModifier

    def __singleLine(self, height: int, width: int, coreRegion: Tuple[int, int]) -> np.ndarray:
        minWidth = int(width * 0.75)
        maxHeight = 10
        corePadding = int((coreRegion[1] - coreRegion[0]) * 0.1)
        x1 = np.random.randint(0, floor(width * 0.1))
        y1 = np.random.randint(coreRegion[0] + corePadding, coreRegion[1] - corePadding)
        x2 = np.random.randint(x1 + minWidth, width)
        y2 = np.random.randint(y1 - int(0.5 * maxHeight), y1 + int(maxHeight * 0.5))
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height
        return np.array([[x1, y1], [x2, y2]])

    def __doubleLine(self, height: int, width: int, coreRegion: Tuple[int, int]) -> np.ndarray:
        firstLine = self.__singleLine(height, width, coreRegion)
        lengthOffset = 10
        minYOffset = floor(height * 0.15)
        maxYOffset = floor(height * 0.35)
        corePad = int((coreRegion[1] - coreRegion[0]) * 0.1)
        x1 = np.random.randint(np.maximum(0, firstLine[0][0] - lengthOffset), firstLine[0][0] + lengthOffset)
        x2 = np.random.randint(firstLine[1][0] - lengthOffset, np.minimum(width, firstLine[1][0] + lengthOffset))
        middleThird = floor(height * 0.3)
        topThird = floor(height * 0.6)
        y_avg = np.mean(firstLine[:, 1])
        y1 = -1
        y2 = -1
        counter = 0
        while (y1 < coreRegion[0] - corePad or y1 > coreRegion[1] + corePad or y2 < coreRegion[
            0] - corePad or y2 > coreRegion[1] + corePad):
            if y_avg < middleThird:
                y1 = firstLine[0, 1] + np.random.randint(minYOffset, maxYOffset)
                y2 = firstLine[1, 1] + np.random.randint(minYOffset, maxYOffset)
            elif y_avg > topThird:
                y1 = firstLine[0, 1] - np.random.randint(minYOffset, maxYOffset)
                y2 = firstLine[1, 1] - np.random.randint(minYOffset, maxYOffset)
            else:
                above = bool(np.random.randint(2))
                if above:
                    y1 = firstLine[0, 1] + np.random.randint(minYOffset, maxYOffset)
                    y2 = firstLine[1, 1] + np.random.randint(minYOffset, maxYOffset)
                else:
                    y1 = firstLine[0, 1] - np.random.randint(minYOffset, maxYOffset)
                    y2 = firstLine[1, 1] - np.random.randint(minYOffset, maxYOffset)
            counter += 1
            if counter > 5:
                # to avoid getting stuck for ever in some kind of unforseeable edge case: just take what we got after 5 tries
                logging.debug('double line gen exited with potentially(!) less than optimal y values '
                              '(reason: exceeded loop limit)')
                break
        if x2 > width:
            x2 = width
        if y1 < 0:
            y1 = 0
        if y2 > height:
            y2 = height
        lineCoordinates = np.append(firstLine, [[x1, y1], [x2, y2]], axis=0)
        return lineCoordinates

    def __cross(self, height: int, width: int, coreRegion: Tuple[int, int] = None) -> np.ndarray:
        minWidth = int(width * 0.75)
        x1 = np.random.randint(0, floor(width * 0.1))
        x2 = np.random.randint(x1 + minWidth, width)
        x3 = np.random.randint(0, floor(width * 0.1))
        x4 = np.random.randint(x3 + minWidth, width)

        y_positions = self.__generateYPositions(height, 4, coreRegion=coreRegion)
        y1 = y_positions[0]
        y2 = y_positions[1]
        y3 = y_positions[3]
        y4 = y_positions[2]

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    def __diagonal(self, height: int, width: int, coreRegion: Tuple[int, int], pad: int = 10) -> np.ndarray:
        min_width = int(width * 0.75)

        yPositions = self.__generateYPositions(height, 2, coreRegion=coreRegion)
        y1 = yPositions[0]
        y2 = yPositions[1]

        x1 = np.random.randint(0, floor(width * 0.1))
        x2 = np.random.randint(x1 + min_width, width)
        return np.array([[x1, y1], [x2, y2]])

    def __wave(self, height: int, width: int, minXStep: int = 10, maxXStep: int = 40, margin: int = 10,
               k: int = 2, coreRegion: Tuple[int, int] = None) -> np.ndarray:
        x = self.__generateXPositions(width, minXStep, maxXStep, margin)
        counter = 0
        while len(x) <= k:
            if counter > 2:
                x = self.__generateXPositions(width, max_step=floor(width / k),
                                              margin=margin)  # width should be > 60 and k==2 or 3
                counter += 1
            elif counter > 6:
                offset = k + 2
                x = [offset * i for i in range(1, k + 2)]
                logging.warning(
                    'wave gen issue: failed to generate sufficient x positions for image width {}, fell back to'
                    ' evenly spaced points'.format(width))
            else:
                x = self.__generateXPositions(width, minXStep, maxXStep, margin)
                counter += 1

        y = self.__generateYPositions(height, len(x), margin=margin, coreRegion=coreRegion)
        tck = interpolate.splrep(x, y, k=k)
        spl = interpolate.BSpline(tck[0], tck[1], tck[2])
        xx = np.linspace(0, width, width * 2)
        yy = spl(xx)
        points = np.array(list(zip(xx, yy)), dtype=np.int64)
        pointFilter = [False if p[0] < x[0] or p[0] > x[-1] else True for p in points]
        points = points[pointFilter]
        return np.array(points)

    def __generateXPositions(self, width: int, min_step: int = 10, max_step: int = 50, margin: int = 5) -> List[int]:
        positions = [np.random.randint(margin, margin + min_step)]

        while positions[-1] < (width - max_step):
            positions.append(positions[-1] + np.random.randint(min_step, max_step))
        return positions

    def __generateYPositions(self, height: int, count: int, coreRegion: Tuple[int, int] = None, margin: int = 10) -> List[
        int]:
        if coreRegion:
            diff = (coreRegion[1] - coreRegion[0])
            core_center = floor(coreRegion[1] - diff * 0.5)
        else:
            core_center = floor(height * 0.5)

        if height <= margin:
            margin = 0

        positions = [np.random.randint(margin, height - margin)]

        for _ in range(count - 1):
            if positions[-1] > core_center:  # next point should be in bottom half
                if core_center <= margin:
                    positions.append(np.random.randint(0, core_center))
                else:
                    positions.append(np.random.randint(margin, core_center))
            else:  # next point should be in top half
                if core_center >= height - margin:
                    positions.append(np.random.randint(core_center, height))
                else:
                    positions.append(np.random.randint(core_center, height - margin))
        return positions

    def __zigZag(self, height: int, width: int, minXStep: int = 10, maxXStep: int = 40,
                 core_region: Tuple[int, int] = None) -> np.ndarray:
        return self.__wave(height, width, minXStep, maxXStep, k=1, coreRegion=core_region)

    def __scratch(self, height: int, width: int, coreRegion: Tuple[int, int], maxScratchCount: int = 30, pad: int = 10,
                  yOffset: int = 10) -> List[List[int]]:
        lineSegments = []
        previousX = np.random.randint(pad, width)
        if coreRegion[0] <= pad:
            previousY = np.random.randint(0, max(coreRegion[0], 1))
        else:
            previousY = np.random.randint(max(pad, coreRegion[0] - pad),
                                          coreRegion[0])  # let's put the first stroke somewhere in the core region

        center = width * 0.5

        lineSegments.append([previousX, previousY])
        for i in range(maxScratchCount):
            if previousX < center:
                nextX = np.random.randint(int(center) + pad, width - pad)
            else:
                nextX = np.random.randint(pad, int(
                    center) - pad)  # pad around the center, to ensure that we have some form of offset

            nextY = previousY + np.random.randint(-2, yOffset)
            if nextY > height - pad:
                return lineSegments
            lineSegments.append([nextX, nextY])
            previousX = nextX
            previousY = nextY
        return lineSegments
