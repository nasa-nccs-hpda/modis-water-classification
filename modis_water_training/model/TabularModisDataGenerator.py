import copy
import datetime
import glob
import os
import warnings
import time

import numpy as np
import pandas as pd

from osgeo import gdal
from osgeo import gdalconst


# -------------------------------------------------------------------------
# TabularModisDataGenerator
#
# Takes MODIS data (HDF format) and extracts tabular data where each
# row is a pixel that is water or land. Data is filtered given QA bit
# parameters.
# -------------------------------------------------------------------------
class TabularModisDataGenerator(object):

    VAL_LOWER_BOUND = -100
    VAL_UPPER_BOUND = 16000
    INPUT_OUTPUT_DIMENSION = 4800
    SENSOR_ZENITH_UPPER_BOUND = 45
    SOLAR_ZENITH_UPPER_BOUND = 67
    LAND_AND_CHECK = 65535
    WATER_AND_CHECK = 1
    PER_DAY_ROW_LIMIT = 2000000
    FILE_PRE_STR = 'HDF4_EOS:EOS_GRID:'

    def __init__(self,
                 tile,
                 year,
                 tileDir='.',
                 outDir='.',
                 waterMaskDir='.',
                 julianDays=None,
                 numDataPoints=100000,
                 excludeDays=None,
                 rm_csv=True,
                 rm_tifs=True,
                 logger=None,
                 client=None):
        self._tile = tile
        self._tileDir = tileDir
        self._outDir = outDir
        self._waterMaskDir = waterMaskDir
        self._numDataPoints = numDataPoints
        self._julienDays = julianDays if julianDays else range(1, 366)
        if excludeDays:
            self._julienDays = [
                day for day in self._julienDays if day not in excludeDays]
        self._logger = logger
        self._year = str(year)
        self._rmcsv = rm_csv
        self._rmTif = rm_tifs
        self._finalCount = 0
        self._client = client

        self._SENZ = 'SensorZenith_1'
        self._SOLZ = 'SolarZenith_1'
        self._SR1 = 'sur_refl_b01_1'
        self._SR2 = 'sur_refl_b02_1'
        self._SR3 = 'sur_refl_b03_1'
        self._SR4 = 'sur_refl_b04_1'
        self._SR5 = 'sur_refl_b05_1'
        self._SR6 = 'sur_refl_b06_1'
        self._SR7 = 'sur_refl_b07_1'
        self._STATE = 'state_1km_1'
        self._NDVI = 'ndvi'
        self._NDWI1 = 'ndwi1'
        self._NDWI2 = 'ndwi2'
        self._XOFF = 'x_offset'
        self._YOFF = 'y_offset'
        self._YEAR_COL = 'year'
        self._JD = 'julian_day'
        self._SRS = [self._SR1, self._SR2, self._SR3, self._SR4,
                     self._SR5, self._SR6, self._SR7]
        self._QAS = [self._STATE, self._SENZ, self._SOLZ]
        self._IDXS = [self._XOFF, self._YOFF]
        self._SRSPLIDX = copy.deepcopy(self._SRS)
        self._SRSPLIDX.append(self._XOFF)
        self._SRSPLIDX.append(self._YOFF)
        self._SRSPLIDX.append(self._NDVI)
        self._SRSPLIDX.append(self._NDWI1)
        self._SRSPLIDX.append(self._NDWI2)
        self._SRSPLIDX.append(self._YEAR_COL)
        self._SRSPLIDX.append(self._JD)
        # ---
        # Bits to check for
        # aerosol = 0b11000000
        # cl_cloudy = 0b1
        # cl_mixed = 0b10
        # cld_shadow = 0b100
        # int_cld = 0b10000000000
        # ---
        self._aerosol = 192
        self._cl_cloudy = 1
        self._cl_mixed = 2
        self._cld_shadow = 4
        self._int_cld = 1024

    # -------------------------------------------------------------------------
    # _generateData
    # -------------------------------------------------------------------------
    def _generateData(self, numRows=10000):
        parquetPaths = []
        timeTotal = []
        timeTotalStr = []
        for day in self._julienDays:
            print('Looking for day {}'.format(day))
            time_start = time.time()
            files = self._readFiles(sensorDir=self._tileDir, julianDay=day)
            if files:
                self._files = files
                parquetPaths.append(self._generateDataPerDay(julianDay=day))
            else:
                print('There are no files for that day.')
            time_end = time.time()
            time_ = float(time_end - time_start)
            time_ = round(time_, 2)
            timeTotal.append(time_)
            timeTotalStr.append('{}: {}\n'.format(day, str(time_)))
        self._logTimes(timeTotal=timeTotal, timeTotalStr=timeTotalStr)
        self._writeMainParquet(parquetPaths, numRows=numRows)

    # -------------------------------------------------------------------------
    # _readFiles
    # -------------------------------------------------------------------------
    def _readFiles(self, julianDay, sensorDir):

        # This is a dictionary mapping the band name to its file.
        files = {}

        pattern = '*GA.A' + \
                  str(self._year) + \
                  str(julianDay).zfill(3) + \
                  '.' + \
                  self._tile + \
                  '*.hdf'

        globDir = os.path.join(sensorDir)
        gaDays = glob.glob(os.path.join(globDir, pattern))
        print(os.path.join(globDir, pattern))

        for gaFile in gaDays:
            print(gaFile)

            bands = [':MODIS_Grid_1km_2D:SensorZenith_1',
                     ':MODIS_Grid_1km_2D:SolarZenith_1',
                     ':MODIS_Grid_1km_2D:state_1km_1',
                     ':MODIS_Grid_500m_2D:sur_refl_b03_1',
                     ':MODIS_Grid_500m_2D:sur_refl_b04_1',
                     ':MODIS_Grid_500m_2D:sur_refl_b05_1',
                     ':MODIS_Grid_500m_2D:sur_refl_b06_1',
                     ':MODIS_Grid_500m_2D:sur_refl_b07_1']

            # ---
            # Get the GQ name.  The last part of the GA name is the production
            # time stamp.  The GQ mate can have a different time stamp, so
            # remove it and glob again.
            # ---
            gqPattern = gaFile.replace('GA', 'GQ')
            gqPattern = os.path.splitext(gqPattern)[0]
            gqPattern = '.'.join(gqPattern.split('.')[0:-1]) + '*.hdf'

            gqFiles = glob.glob(gqPattern)

            try:
                if len(gqFiles) == 0:

                    raise RuntimeError('No GQ file found for ' + gaFile)

                elif len(gqFiles) > 1:

                    raise RuntimeError('Found more than one GQ file with ' +
                                       gqPattern)

                files.update(self._readBands(gaFile, bands))
                bands = [':MODIS_Grid_2D:sur_refl_b01_1',
                         ':MODIS_Grid_2D:sur_refl_b02_1']

                files.update(self._readBands(gqFiles[0], bands))
            except RuntimeError:
                print('Julian day: {}\nFiles: {}'.format(julianDay, files))
                print('Could not find files for that day, skipping.')

        print('Julian day: {}\nFiles: {}'.format(julianDay, files))
        return files

    # -------------------------------------------------------------------------
    # _readBands
    # -------------------------------------------------------------------------
    def _readBands(self, hdfFile, bands):
        bandFiles = {}
        for band in bands:
            bandFile = self.readFile(hdfFile, band)
            if band not in bandFiles:
                # Remove the resolution string from the band name.
                bandFiles[band.split(':')[-1]] = bandFile
        return bandFiles

    # -------------------------------------------------------------------------
    # readFile
    # -------------------------------------------------------------------------
    def readFile(self, dayFile, band):
        outName = '{}"{}"{}'.format(self.FILE_PRE_STR, dayFile, band)
        return outName

    # -------------------------------------------------------------------------
    # _generateDataPerDay
    # -------------------------------------------------------------------------
    def _generateDataPerDay(self, julianDay):
        maskForDay, _ = self._generateBadDataMask(day=julianDay, 
                                                  files=self._files)
        waterMask = self._readWaterMask()
        waterConditional, landConditional = self._generateWaterLandConditional(
            waterMask,
            maskForDay)
        waterDict, landDict = self._extractBandsPerConditional(
            waterConditional, landConditional)
        waterDict, landDict = self._extractIndices(
            waterConditional, landConditional, waterDict, landDict
        )
        waterDict = self._addIndices(waterDict)
        landDict = self._addIndices(landDict)
        waterDict = self._addYearDay(julianDay, waterConditional, waterDict)
        landDict = self._addYearDay(julianDay, landConditional, landDict)
        waterMatrix = self._bandsToMatrix(waterDict)
        landMatrix = self._bandsToMatrix(landDict)
        waterCategorical = self._generateCategorical(
            waterMatrix.shape, water=True)
        landCategorical = self._generateCategorical(
            landMatrix.shape, water=False)
        waterAll = self._hstackCategorical(waterCategorical, waterMatrix)
        landAll = self._hstackCategorical(landCategorical, landMatrix)
        waterCSVPath = self._writeMatrixToParquet(
            waterAll, day=julianDay, water=True)
        landCSVPath = self._writeMatrixToParquet(
            landAll, day=julianDay, water=False)
        return (waterCSVPath, landCSVPath)

    # -------------------------------------------------------------------------
    # _generateBadDataMask
    # -------------------------------------------------------------------------
    def _generateBadDataMask(self, day, files):
        npArrayBands = {}
        for qaBand in self._QAS:
            npArrayBands[qaBand] = self._fileToArray(
                files[qaBand])

        zeros_matrix = np.zeros((self.INPUT_OUTPUT_DIMENSION,
                                 self.INPUT_OUTPUT_DIMENSION))
        ones_matrix = np.ones((self.INPUT_OUTPUT_DIMENSION,
                               self.INPUT_OUTPUT_DIMENSION))
        cond = self._generateConditionalArray(npArrayBands)
        ndMask = np.where(cond, ones_matrix, zeros_matrix).astype(np.uint16)
        cond = None
        outPath = self._writeBadDataMask(mask=ndMask, day=day)
        return ndMask,outPath

    # -------------------------------------------------------------------------
    # _writeBadDataMask
    # -------------------------------------------------------------------------
    def _writeBadDataMask(self, mask, day):
        post_str = self._getPostStr()
        #uncomment to include a timestamp for the QA mask
        #outName = 'MOD.A{}{:03}.{}.{}.QA.tif'.format(
        #    self._year, day, self._tile, post_str)
        outName = 'MOD.A{}{:03}.{}.{}.QA.tif'.format(
            self._year, day, self._tile, post_str)
        outPath = os.path.join(self._outDir, outName)
        driver = gdal.GetDriverByName('GTiff')
        outDs = driver.Create(outPath,
                              self.INPUT_OUTPUT_DIMENSION,
                              self.INPUT_OUTPUT_DIMENSION,
                              1,
                              gdalconst.GDT_UInt16)
        outDs.SetGeoTransform(self._outTransform)
        outDs.SetProjection(self._outProjection)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(mask)
        outBand.FlushCache()
        outBand = None
        outDs = None
        mask = None
        return outPath

    # -------------------------------------------------------------------------
    # _generateConditionalArray
    # -------------------------------------------------------------------------
    def _generateConditionalArray(self, bandArrays):
        state = bandArrays[self._STATE]
        cld_condition0 = ((state & self._cld_shadow) == self._cld_shadow)
        cld_condition1 = ((state & self._cl_cloudy) == self._cl_cloudy)
        cld_condition2 = ((state & self._cl_mixed) == self._cl_mixed)
        aerosolCondition = ((state & self._aerosol) == self._aerosol)
        sensor_condition = ((bandArrays[self._SENZ]/100) >=
                            self.SENSOR_ZENITH_UPPER_BOUND)
        solar_condition = ((bandArrays[self._SOLZ]/100) >=
                           self.SOLAR_ZENITH_UPPER_BOUND)
        cldCondition = (cld_condition0 | cld_condition1 | cld_condition2)
        sensSolCondition = (sensor_condition | solar_condition)
        totalCondition = (cldCondition | aerosolCondition | sensSolCondition)
        if not totalCondition.any():
            msg = 'No valid pixels'
            warnings.warn(msg)
        if totalCondition.all():
            msg = 'Something weird happened, it should not be this good'
            warnings.warn(msg)
        return totalCondition

    # -------------------------------------------------------------------------
    # _fileToArray
    # -------------------------------------------------------------------------
    def _fileToArray(self, name, type_=np.uint16):
        ds = gdal.Open(name)
        if self._STATE in name:
            self._outTransform = ds.GetGeoTransform()
            self._outProjection = ds.GetProjection()
        bandNdarray = ds.ReadAsArray(0, 0, None, None, None,
                                     self.INPUT_OUTPUT_DIMENSION,
                                     self.INPUT_OUTPUT_DIMENSION)
        bandNdarray = bandNdarray.astype(type_)
        return bandNdarray

    # -------------------------------------------------------------------------
    # _readWaterMask
    # -------------------------------------------------------------------------
    def _readWaterMask(self):
        pattern = '*{}*.tif'.format(self._tile)
        waterMasks = glob.glob(os.path.join(self._waterMaskDir, pattern))
        if len(waterMasks) == 0:
            msg = 'No water mask found at {}'.format(
                os.path.join(self._waterMaskDir, pattern))
            raise RuntimeError(msg)
        else:
            return self._fileToArray(waterMasks[0])

    # -------------------------------------------------------------------------
    # _generateWaterLandConditional
    # -------------------------------------------------------------------------
    def _generateWaterLandConditional(self, waterMask, badDataMask):
        waterConditional = ((~badDataMask & waterMask) == self.WATER_AND_CHECK)
        landConditional = ((~badDataMask & ~waterMask) == self.LAND_AND_CHECK)
        waterMask = None
        badDataMask = None
        return waterConditional, landConditional

    # -------------------------------------------------------------------------
    # _extractBandsPerConditional
    # -------------------------------------------------------------------------
    def _extractBandsPerConditional(self, waterConditional, landConditional):

        waterBandArrayDict = {}
        landBandArrayDict = {}

        for band in self._SRS:
            if not band == self._SRS[0]:
                bandArray = self._fileToArray(self._files[band],
                                              type_=np.int16)
            else:
                bandArray = self._fileToArray(self._files[band],
                                              type_=np.int16)
            waterExtract = np.extract(waterConditional, bandArray)
            landExtract = np.extract(landConditional, bandArray)
            waterBandArrayDict[band] = waterExtract
            landBandArrayDict[band] = landExtract

            # Free up some memory.
            bandArray = None
            waterExtract = None
            landExtract = None

        return waterBandArrayDict, landBandArrayDict

    # -------------------------------------------------------------------------
    # _extractIndices
    # -------------------------------------------------------------------------
    def _extractIndices(self,
                        waterConditional,
                        landConditional,
                        waterBandArrayDict,
                        landBandArrayDict):
        shape = self._fileToArray(
            self._files[self._SRS[0]], type_=np.int16).shape
        idxMat = np.indices(shape)
        x_off = idxMat[:][:][1]
        y_off = idxMat[:][:][0]
        water_x_extract = np.extract(waterConditional, x_off)
        water_y_extract = np.extract(waterConditional, y_off)
        land_x_extract = np.extract(landConditional, x_off)
        land_y_extract = np.extract(landConditional, y_off)
        waterBandArrayDict[self._XOFF] = water_x_extract
        waterBandArrayDict[self._YOFF] = water_y_extract
        landBandArrayDict[self._XOFF] = land_x_extract
        landBandArrayDict[self._YOFF] = land_y_extract

        # Don't need these anymore.
        waterConditional = None
        landConditional = None
        idxMat = None
        x_off = None
        y_off = None
        water_x_extract = None
        water_y_extract = None
        land_x_extract = None
        land_y_extract = None
        return waterBandArrayDict, landBandArrayDict

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
    # _bandsToMatrix
    # -------------------------------------------------------------------------
    def _bandsToMatrix(self, bandDict):
        for i, key in enumerate(bandDict.keys()):
            if i == 0:
                fullMatrix = bandDict[key]
            else:
                fullMatrix = np.vstack(
                    (fullMatrix, bandDict[key])).astype(np.float32)
        fullMatrix = np.transpose(fullMatrix)
        bandDict = None
        return fullMatrix

    # -------------------------------------------------------------------------
    # _generateCategorical
    # -------------------------------------------------------------------------
    def _generateCategorical(self, shape, water=True):
        waterLandCategorical = np.zeros((shape[0], 1))
        if water:
            waterLandCategorical[:, 0] = np.float32(1)
        return waterLandCategorical

    # -------------------------------------------------------------------------
    # _hstackCategorical
    # -------------------------------------------------------------------------
    def _hstackCategorical(self, categorical, matrix):
        stackedMatrix = np.hstack((categorical, matrix)).astype(np.float32)
        np.random.shuffle(stackedMatrix)
        categorical = None
        matrix = None
        return stackedMatrix

    # -------------------------------------------------------------------------
    # _writeMatrixToParquet
    # -------------------------------------------------------------------------
    def _writeMatrixToParquet(self, matrix, day, water=True):
        fields = ['water']
        fields.extend(self._SRSPLIDX)
        landWater = 'water' if water else 'land'
        matrix = matrix[:][:self.PER_DAY_ROW_LIMIT]
        shape = matrix.shape
        post_str = self._getPostStr()
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
    # _logTimes
    # -------------------------------------------------------------------------
    def _logTimes(self, timeTotal, timeTotalStr):
        path = os.path.join(self._outDir, 'jobTimeLogged{}{}.log'.format(
            self._year, self._tile))
        with open(path, 'w') as time_log:
            time_log.write('#########################\n')
            time_log.write('{}'.format(datetime.datetime.now()))
            time_log.write('\nTimes per day\n')
            time_log.writelines(timeTotalStr)
            time_log.write('Average: {} sec\n'.format(
                sum(timeTotal)/len(timeTotal)))
            time_log.write('Total time: {} sec\n'.format(sum(timeTotal)))
            time_log.write('Total days: {}\n'.format(len(timeTotal)))
            time_log.write('#########################\n\n')

    # -------------------------------------------------------------------------
    # _writeMainCSV
    # -------------------------------------------------------------------------
    def _writeMainParquet(self, pathList, numRows=10000):
        dataframeConcat = TabularModisDataGenerator.readParquetChunked(
            pathList, self._rmcsv)
        dataframeConcat = TabularModisDataGenerator.filterNegativeRows(
            dataframeConcat)
        self._finalCount = len(dataframeConcat.index)
        print('Final count: {}'.format(self._finalCount))
        post_str = self._getPostStr()
        outputPath = 'MOD.A{}.{}.{}.{}.all.parquet.gzip'.format(
            self._year,
            self._tile,
            post_str,
            self._finalCount
        )
        outputPath = os.path.join(self._outDir, outputPath)
        print('Main parquet: {}'.format(outputPath))
        dataframeConcat = dataframeConcat.sample(frac=1)
        dataframeConcat = dataframeConcat.reset_index()
        dataframeConcat = dataframeConcat.drop(['index'], axis=1)
        dataframeConcat.to_parquet(outputPath)

    # -------------------------------------------------------------------------
    # readCSVChuncked
    # -------------------------------------------------------------------------
    @staticmethod
    def readParquetChunked(pathList, rmCSV):
        dataframeList = []
        for pathTuple in pathList:
            size0 = int(pathTuple[0].split('-')[1])
            size1 = int(pathTuple[1].split('-')[1])
            print('Size 0: {}'.format(size0))
            print('Size 0: {}'.format(size1))
            chunksizeCurrent = size0 if size0 < size1 else size1
            for path in pathTuple:
                print('Reading in {}.\nChunk: {} rows\n'.format(
                    path, chunksizeCurrent))
                df = pd.read_parquet(path)
                df = df.sample(n=chunksizeCurrent)
                dataframeList.append(df)
                if rmCSV:
                    os.remove(path)
        dataframeConcat = pd.concat(dataframeList)
        dataframeList = None
        return dataframeConcat

    # -------------------------------------------------------------------------
    # _filterNegativeRows
    # -------------------------------------------------------------------------
    @staticmethod
    def filterNegativeRows(dataframe):
        for column in dataframe.columns.tolist()[1:]:
            if 'sur_refl' in column:
                print(column)
                dataframe = dataframe[dataframe[column] >=
                                      TabularModisDataGenerator.VAL_LOWER_BOUND]
                dataframe = dataframe[dataframe[column] <=
                                      TabularModisDataGenerator.VAL_UPPER_BOUND]
        return dataframe

    # -------------------------------------------------------------------------
    # _getPostStr()
    # -------------------------------------------------------------------------
    def _getPostStr(self):
        sdtdate = datetime.datetime.now()
        year = sdtdate.year
        hm = sdtdate.strftime('%H%M')
        sdtdate = sdtdate.timetuple()
        jdate = sdtdate.tm_yday
        post_str = '{}{:03}{}'.format(year, jdate, hm)
        return post_str

    # -------------------------------------------------------------------------
    # run()
    # -------------------------------------------------------------------------
    def run(self):
        return self._generateData()
