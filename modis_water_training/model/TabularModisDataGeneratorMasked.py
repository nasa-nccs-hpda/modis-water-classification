import csv
import os
import time

import numpy as np
import pandas as pd

from modis_water_random_forest.model.TabularModisDataGenerator import TabularModisDataGenerator


# -------------------------------------------------------------------------
# TabularModisDataGeneratorMasked
#
# Similar functionality to TabularModisDataGeneratorMasked except the mask
# used is a user-given mask instead of a water mask. Only one type of pixel
# is extracted (land or water).
# -------------------------------------------------------------------------
class TabularModisDataGeneratorMasked(TabularModisDataGenerator):

    EXTRACTION_VALUE = 0

    def __init__(self,
                 tile,
                 year,
                 maskPath,
                 water=True,
                 tileDir='.',
                 outDir='.',
                 julianDays=None,
                 numDataPoints=100000,
                 excludeDays=None,
                 rm_csv=True,
                 logger=None,
                 noQA=False):
        print(julianDays)
        super(TabularModisDataGeneratorMasked, self).__init__(
            tile,
            year,
            tileDir=tileDir,
            outDir=outDir,
            julianDays=julianDays,
            numDataPoints=numDataPoints,
            excludeDays=excludeDays,
            rm_csv=rm_csv,
            logger=logger
        )

        print(self._julienDays)
        self._noQA = noQA
        print('QA mask being applied: {}'.format(
            False if self._noQA else True))
        self._maskPath = maskPath
        self._water = water

    # -------------------------------------------------------------------------
    # _generateData
    # -------------------------------------------------------------------------
    def _generateData(self):
        parquetPaths = []
        timeTotal = []
        timeTotalStr = []
        for day in self._julienDays:
            print('Looking for day {}'.format(day))
            time_start = time.time()
            files = self._readFiles(julianDay=day, sensorDir=self._tileDir)
            if files:
                self._files = files
                print()
                parquetPaths.append(self._generateDataPerDay(day=day))
            else:
                print('There are no files for that day.')
            time_end = time.time()
            time_ = float(time_end - time_start)
            time_ = round(time_, 2)
            timeTotal.append(time_)
            timeTotalStr.append('{}: {}\n'.format(day, str(time_)))
        self._logTimes(timeTotal=timeTotal, timeTotalStr=timeTotalStr)
        self._writeMainParquet(parquetPaths, numRows=100000)

    # -------------------------------------------------------------------------
    # _generateDataPerDay
    # -------------------------------------------------------------------------
    def _generateDataPerDay(self, day):
        maskForDay = self._generateBadDataMask(day=day)
        mask = self._readMask()
        maskConditional = self._generateConditional(mask, maskForDay)
        bandDict = self._extractBandsPerConditional(maskConditional)
        bandDict = self._extractIndices(maskConditional, bandDict)
        bandDict = self._addIndices(bandDict)
        bandDict = self._addYearDay(day, maskConditional, bandDict)
        matrix = self._bandsToMatrix(bandDict)
        categorical = self._generateCategorical(matrix.shape,
                                                water=self._water)
        landAll = self._hstackCategorical(categorical, matrix)
        parquetPath = self._writeMatrixToParquet(landAll, day=day)
        print('Parquet path: {}'.format(parquetPath))
        return parquetPath

    # -------------------------------------------------------------------------
    # _readMask
    # -------------------------------------------------------------------------
    def _readMask(self):
        print('Reading mask from: {}'.format(self._maskPath))
        if not os.path.exists(self._maskPath):
            msg = 'No mask found at: {}'.format(self._maskPath)
            raise RuntimeError(msg)
        else:
            return self._fileToArray(self._maskPath)

    # -------------------------------------------------------------------------
    # _extractBandsPerConditional
    # -------------------------------------------------------------------------
    def _extractBandsPerConditional(self, conditional):
        bandArrayDict = {}
        for band in self._SRS:
            bandArray = self._fileToArray(self._files[band], type_=np.int16)
            print(band)
            extract = np.extract(conditional, bandArray)
            bandArrayDict[band] = extract
            # Free up some memory.
            bandArray = None
        return bandArrayDict

    # -------------------------------------------------------------------------
    # _generateConditional
    # -------------------------------------------------------------------------
    def _generateConditional(self, mask, badDataMask):
        print('To-take total in mask:')
        print(np.count_nonzero(mask == 0))
        print('Not to toke total in mask:')
        print(np.count_nonzero(mask == 1))
        cat = (mask == 0)
        extract = np.extract(cat, badDataMask)
        print('Bad data in shape: {}'.format(np.count_nonzero(extract == 1)))
        print('Categorical testing')
        c1 = (~badDataMask & mask)
        print('N out of mask: {}'.format(np.count_nonzero(c1 == 1)))
        c2 = (~badDataMask & ~mask)
        print('N in mask: {}'.format(np.count_nonzero(c2 == 65535)))
        badDataMask = np.zeros(badDataMask.shape, dtype=np.uint16) if \
            self._noQA else badDataMask
        conditional = ((~badDataMask & ~mask) == 65535)
        badDataMask = None
        return conditional

    # -------------------------------------------------------------------------
    # _extractIndices
    # -------------------------------------------------------------------------
    def _extractIndices(self,
                        conditional,
                        bandArrayDict):
        shape = self._fileToArray(
            self._files[self._SRS[0]], type_=np.int16).shape
        idxMat = np.indices(shape)
        x_off = idxMat[:][:][1]
        y_off = idxMat[:][:][0]
        x_extract = np.extract(conditional, x_off)
        y_extract = np.extract(conditional, y_off)
        bandArrayDict[self._XOFF] = x_extract
        bandArrayDict[self._YOFF] = y_extract

        # Don't need these anymore.
        idxMat = None
        x_off = None
        y_off = None
        return bandArrayDict

    # -------------------------------------------------------------------------
    # _extractIndices
    # -------------------------------------------------------------------------
    def _addYearDay(self,
                    julian_day,
                    conditional,
                    bandArrayDict):
        shape = self._fileToArray(
            self._files[self._SRS[0]], type_=np.int16).shape
        year_mat = np.full(shape=shape, fill_value=int(
            self._year), dtype=np.int16)
        day_mat = np.full(shape=shape, fill_value=int(
            julian_day), dtype=np.int16)
        year_extract = np.extract(conditional, year_mat)
        day_extract = np.extract(conditional, day_mat)
        bandArrayDict[self._YEAR_COL] = year_extract
        bandArrayDict[self._JD] = day_extract

        # Don't need these anymore.
        year_mat = None
        day_mat = None
        return bandArrayDict

    # -------------------------------------------------------------------------
    # _addIndices
    # -------------------------------------------------------------------------

    def _addIndices(self, bandDict):
        ndvi = ((bandDict[self._SR2] - bandDict[self._SR1]) /
                (bandDict[self._SR2] + bandDict[self._SR1])) * 10000
        ndwi1 = ((bandDict[self._SR2] - bandDict[self._SR6]) /
                 (bandDict[self._SR2] + bandDict[self._SR6])) * 10000
        ndwi2 = ((bandDict[self._SR2] - bandDict[self._SR7]) /
                 (bandDict[self._SR2] + bandDict[self._SR7])) * 10000
        bandDict[self._NDVI] = ndvi
        bandDict[self._NDWI1] = ndwi1
        bandDict[self._NDWI2] = ndwi2
        return bandDict

    # -------------------------------------------------------------------------
    # _generateCategorical
    # -------------------------------------------------------------------------
    def _generateCategorical(self, shape, water=True):
        waterLandCategorical = np.zeros((shape[0], 1))
        if water:
            waterLandCategorical[:, 0] = np.int16(1)
        return waterLandCategorical

    # -------------------------------------------------------------------------
    # _bandsToMatrix
    # -------------------------------------------------------------------------
    def _bandsToMatrix(self, bandDict):
        for i, key in enumerate(bandDict.keys()):
            if i == 0:
                fullMatrix = bandDict[key]
            else:
                fullMatrix = np.vstack(
                    (fullMatrix, bandDict[key])).astype(np.int16)
        fullMatrix = np.transpose(fullMatrix)
        bandDict = None
        return fullMatrix

    # -------------------------------------------------------------------------
    # _hstackCategorical
    # -------------------------------------------------------------------------
    def _hstackCategorical(self, categorical, matrix):
        stackedMatrix = np.hstack((categorical, matrix)).astype(np.int16)
        np.random.shuffle(stackedMatrix)
        categorical = None
        matrix = None
        return stackedMatrix

    # -------------------------------------------------------------------------
    # _writeMatrixToCSV
    # -------------------------------------------------------------------------
    def _writeMatrixToParquet(self, matrix, day, shuffle=True):
        fields = ['water']
        fields.extend(self._SRSPLIDX)
        shape = matrix.shape
        post_str = self._getPostStr()
        landWater = 'water' if self._water else 'land'
        parquetPath = 'MOD.A{}{:03}.{}.{}-{}-{}.parquet.gzip'.format(
            self._year, day, self._tile, post_str, shape[0], landWater)
        parquetPath = os.path.join(self._outDir, parquetPath)
        print('Writing matrix to parquet: {}\nRows: {}\n'.format(
            parquetPath, matrix.shape[0]))
        if os.path.exists(parquetPath):
            print('{} already exists.'.format(parquetPath))
            matrix = None
            return parquetPath
        dfFromMatrix = pd.DataFrame(matrix, columns=fields)
        dfFromMatrix.to_parquet(parquetPath)
        matrix = None
        dfFromMatrix = None
        return parquetPath

    # -------------------------------------------------------------------------
    # _writeMainParquet
    # -------------------------------------------------------------------------
    def _writeMainParquet(self, pathList, numRows=10000):
        dfList = [pd.read_parquet(name) for name in pathList]
        dataframeConcat = pd.concat(dfList)
        dataframeConcat = TabularModisDataGenerator.filterNegativeRows(
            dataframeConcat)
        self._finalCount = len(dataframeConcat.index)
        print('Final count: {}'.format(self._finalCount))
        post_str = self._getPostStr()
        outputPath = 'MOD.A{}.{}.{}.{}.{}.all.parquet.gzip'.format(
            self._year,
            self._tile,
            post_str,
            'water' if self._water else 'land',
            self._finalCount
        )
        outputPath = os.path.join(self._outDir, outputPath)
        print('Main Parquet: {}'.format(outputPath))
        dataframeConcat = dataframeConcat.sample(frac=1)
        dataframeConcat = dataframeConcat.reset_index()
        dataframeConcat = dataframeConcat.drop(['index'], axis=1)
        dataframeConcat.to_parquet(outputPath)

    # -------------------------------------------------------------------------
    # run()
    # -------------------------------------------------------------------------
    def run(self):
        return self._generateData()
