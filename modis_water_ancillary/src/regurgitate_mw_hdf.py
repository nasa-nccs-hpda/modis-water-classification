import datetime
import os
import logging

import numpy as np
from osgeo import gdal


MW_WATER_MASK: str = 'water_mask'
MW_QA_MASK: str = 'qa_mask'
SEVEN_CLASS_MASK: str = 'seven_class'
HDF_NAME_PRE_STR: str = 'MOD44W.A'


def write_out(array: np.ndarray,
              projection,
              transform,
              out_dir: str,
              layer_type: str,
              year: int,
              tile: str,
              geotiff: bool = True) -> int:
    output_name = get_name(layer_type, year, tile)
    logging.info(f'Writing out {output_name}')
    write_product(layer_type, out_dir, output_name,
                  array, projection, transform, geotiff)
    return 1


def get_post_str():
    sdtdate = datetime.datetime.now()
    year = sdtdate.year
    hm = sdtdate.strftime('%H%M')
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    post_str = '{}{:03}{}'.format(year, jdate, hm)
    return post_str


def get_name(layer_type: str, year: int, tile: str) -> str:
    return f'{HDF_NAME_PRE_STR}{year}001.{tile}' + \
        f'.061.{layer_type}.{get_post_str()}'


def write_product(outType, outDir, outName, array, projection, transform,
                  geoTiff=True) -> None:
    try:
        cols = array.shape[0]
        rows = array.shape[1] if len(
            array.shape) > 1 else 1
        fileType = '.tif' if geoTiff else '.bin'
        imageName = os.path.join(outDir, outName + fileType)
        logging.info(f'Writing to {imageName}')
        driver = gdal.GetDriverByName('GTiff') if geoTiff \
            else gdal.GetDriverByName('ENVI')
        options = ['COMPRESS=LZW'] if geoTiff else []
        ds = driver.Create(imageName, cols, rows, 1, gdal.GDT_Byte,
                           options=options)
        if projection:
            ds.SetProjection(projection)
        if transform:
            ds.SetGeoTransform(transform)

        band = ds.GetRasterBand(1)
        band.WriteArray(array, 0, 0)
        band = None
        ds = None
        logging.info(
            'Wrote annual {} products to: {}'.format(outType,
                                                     imageName))
    except Exception as e:
        msg = f'Encountered {str(e)} during write out using GDAL'
        raise RuntimeError(msg)
    return None
