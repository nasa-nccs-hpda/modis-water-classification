import argparse
import glob
import logging
import os
import sys

from osgeo import gdal
import numpy as np

COLS: int = 4800
ROWS: int = COLS

HDF: str = '.hdf'

# Bands
SENZ: str = 'SensorZenith_1'
SOLZ: str = 'SolarZenith_1'
SR1: str = 'sur_refl_b01_1'
SR2: str = 'sur_refl_b02_1'
SR3: str = 'sur_refl_b03_1'
SR4: str = 'sur_refl_b04_1'
SR5: str = 'sur_refl_b05_1'
SR6: str = 'sur_refl_b06_1'
SR7: str = 'sur_refl_b07_1'
STATE: str = 'state_1km_1'

GA_BANDS: set = set([SENZ, SOLZ, SR3, SR4, SR5, SR6, SR7, STATE])
GQ_BANDS: set = set([SR1, SR2])
ALL_BANDS: set = GA_BANDS | GQ_BANDS

# Sensors
MOD: str = 'MOD'
MYD: str = 'MYD'
SENSORS: set = set([MOD, MYD])


# -------------------------------------------------------------------------
# getBandName
# -------------------------------------------------------------------------
def getBandName(bandFile: str) -> str:

    band = os.path.splitext(os.path.basename(bandFile))[0]. \
        split('-')[1]

    return band


# -------------------------------------------------------------------------
# tanslate
# -------------------------------------------------------------------------
def translate(hdfRegex: str, bandType: str, outDir: str) -> None:

    # Do we need GA files, GQ files, or both?
    gaBands = GA_BANDS
    gqBands = GQ_BANDS

    if bandType == 'GA':
        logging.info('Processing GA')
        hdfFiles = glob.glob(hdfRegex)
        writeBandsFromHdfs(hdfFiles, gaBands, outDir)

    if bandType == 'GQ':
        logging.info('Processing GQ')
        hdfFiles = glob.glob(hdfRegex)
        writeBandsFromHdfs(hdfFiles, gqBands, outDir)

    return None


# -------------------------------------------------------------------------
# writeBandsFromHdfs
# -------------------------------------------------------------------------
def writeBandsFromHdfs(hdfFiles: list, bands: set, outDir: str) \
        -> None:

    # ---
    # We use abbreviated band names elsewhere, but we must use full names
    # now.
    # ---
    FULL_BAND_NAMES = {
        SENZ: ':MODIS_Grid_1km_2D:SensorZenith_1',
        SOLZ: ':MODIS_Grid_1km_2D:SolarZenith_1',
        SR1: ':MODIS_Grid_2D:sur_refl_b01_1',
        SR2: ':MODIS_Grid_2D:sur_refl_b02_1',
        SR3: ':MODIS_Grid_500m_2D:sur_refl_b03_1',
        SR4: ':MODIS_Grid_500m_2D:sur_refl_b04_1',
        SR5: ':MODIS_Grid_500m_2D:sur_refl_b05_1',
        SR6: ':MODIS_Grid_500m_2D:sur_refl_b06_1',
        SR7: ':MODIS_Grid_500m_2D:sur_refl_b07_1',
        STATE: ':MODIS_Grid_1km_2D:state_1km_1'
    }
    for hdfFile in hdfFiles:

        for band in bands:

            subDataSet = 'HDF4_EOS:EOS_GRID:"' + \
                hdfFile + '":' + \
                FULL_BAND_NAMES[band]
            logging.info(f'Reading: {subDataSet}')
            ds = gdal.Open(subDataSet)

            xform = ds.GetGeoTransform()
            proj = ds.GetProjection()

            bandArray = ds.ReadAsArray(0, 0, None, None, None,
                                       COLS,
                                       ROWS)
            geotiffName = getName(hdfFile, band)
            writeRaster(outDir, bandArray, geotiffName,
                        COLS, ROWS,
                        projection=proj,
                        transform=xform)

    return None


# -------------------------------------------------------------------------
# getName
# -------------------------------------------------------------------------
def getName(hdfFileName: str, bandName: str) -> str:
    baseName = os.path.basename(hdfFileName).replace(HDF, '')
    return f'{baseName}.{bandName}'


# -------------------------------------------------------------------------
# writeRaster
# -------------------------------------------------------------------------
def writeRaster(outDir: str, pixels: np.ndarray, name: str,
                cols: int = 0, rows: int = 0, projection=None,
                transform=None) -> None:

    cols = pixels.shape[0]
    rows = pixels.shape[1] if len(pixels.shape) > 1 else 1
    imageName = os.path.join(outDir, name + '.tif')
    logging.info(f'Writing to: {imageName}')
    driver = gdal.GetDriverByName('GTiff')

    ds = driver.Create(imageName, cols, rows, 1, gdal.GDT_Int16,
                       options=['COMPRESS=LZW'])
    if projection:
        ds.SetProjection(projection)
    if transform:
        ds.SetGeoTransform(transform)

    ds.WriteRaster(0, 0, cols, rows, pixels.tobytes())


def main() -> None:
    # Process command-line args.
    desc = 'Use this application to export MODIS HDF as GeoTiff.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-mod',
                        required=True,
                        help='MODIS Regex for GA and GQ products')

    parser.add_argument('-bands',
                        required=True,
                        help='Which product to output: GA or GQ')

    parser.add_argument('-o',
                        default='.',
                        help='Output directory')
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    translate(hdfRegex=args.mod, bandType=args.bands, outDir=args.o)


if __name__ == '__main__':
    sys.exit(main())
