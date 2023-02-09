import argparse
import matplotlib.pyplot as plt
import sys
import logging
import os
import glob
import numpy as np

from osgeo import gdal

from ingest_mw_hdf import ingest_mw_hdf


MW_WATER_MASK: str = 'water_mask'
MW_QA_MASK: str = 'qa_mask'
SEVEN_CLASS_MASK: str = 'seven_class'
LAYER_TYPE_LIST: list = [MW_WATER_MASK, MW_QA_MASK, SEVEN_CLASS_MASK]
ANCILLARY_WATER: int = 0
ANCILLARY_LAND: int = 1
ANCILLARY_OCEAN: int = 2
ANCILLARY_NODATA: int = 10

ANCILLARY_PRE_STR: str = 'Dyn_Water_Ancillary_'
ANCILLARY_POST_STR: str = '_v3b.tif'

def diff_products(year: int,
                  tile: str,
                  hdf_base_path: str,
                  out_dir: str,
                  prod_code: str) -> int:
    hdf_products = ingest_mw_hdf(hdf_base_path, year, tile)
    water_product = sorted(glob.glob(os.path.join(
        out_dir, f'MOD44W.A{year}001.{tile}.061.{MW_WATER_MASK}.{prod_code}.tif')))[0]
    qa_product = sorted(glob.glob(os.path.join(
        out_dir, f'MOD44W.A{year}001.{tile}.061.{MW_QA_MASK}.{prod_code}.tif')))[0]
    sc_product = sorted(glob.glob(os.path.join(
        out_dir, f'MOD44W.A{year}001.{tile}.061.{SEVEN_CLASS_MASK}.{prod_code}.tif')))[0]
    #qa_product = sorted(glob.glob(os.path.join(out_dir, f'MOD44W.A{year}.{tile}.061.{MW_QA_MASK}.*.tif')))[0]
    #seven_product = sorted(glob.glob(os.path.join(out_dir, f'MOD44W.A{year}.{tile}.061.{SEVEN_CLASS_MASK}.*.tif')))[0]
    water_product_dict = read_path(water_product)
    qa_dict = read_path(qa_product)
    sc_dict = read_path(sc_product)
    schdf_unique = np.unique(
        hdf_products[SEVEN_CLASS_MASK]['ndarray'], return_counts=True)
    sc_unique = np.unique(sc_dict['ndarray'], return_counts=True)
    qahdf_unique = np.unique(
        hdf_products[MW_QA_MASK]['ndarray'], return_counts=True)
    qa_unique = np.unique(qa_dict['ndarray'], return_counts=True)
    print(f'Org Seven Class: {schdf_unique}')
    print(f'Mod Seven Class: {sc_unique}')
    print(f'Org QA: {qahdf_unique}')
    print(f'Mod QA: {qa_unique}')
    #qa_product = read_path(water_prod)
    hdf_unique = np.unique(
        hdf_products[MW_WATER_MASK]['ndarray'], return_counts=True)
    mod_water_mask = np.unique(
        water_product_dict['ndarray'], return_counts=True)
    print(f'Org Water Mask: {hdf_unique}')
    print(f'Mod Water Mask: {mod_water_mask}')
    water_diff = hdf_products[MW_WATER_MASK]['ndarray'].astype(
        np.int16) - water_product_dict['ndarray'].astype(np.int16)
    print(np.unique(water_diff, return_counts=True))
    plt.figure(figsize=(15, 15))
    plt.matshow(np.where(water_diff == 255, -2, water_diff), fignum=1)
    # plt.colorbar()
    plt.savefig('water_diff.png')

    plt.matshow(qa_dict['ndarray'], fignum=2)
    plt.colorbar()
    plt.savefig('water_qa.png')
    np.unique(np.where(water_diff == 255, -2, water_diff), return_counts=True)
    write_product('diff', outDir='.', outName=f'MOD44W.A{year}.{tile}.061', array=water_diff,
                  projection=water_product_dict['projection'], transform=water_product_dict['transform'])
    return 0


def read_path(file_path: str) -> dict:
    try:
        ancillary_dataset_object = gdal.Open(file_path)
        ancillary_dataset_band = ancillary_dataset_object.GetRasterBand(1)
        ancillary_dataset_array = ancillary_dataset_band.ReadAsArray()
        ancillary_dataset_transform = \
            ancillary_dataset_object.GetGeoTransform()
        ancillary_dataset_projection = \
            ancillary_dataset_object.GetProjection()
    except Exception as e:
        msg = 'Encountered while trying' + \
            f' to read {file_path} with GDAL'
        raise RuntimeError(str(e) + msg)
    ancillary_dataset_dict = {}
    ancillary_dataset_dict['ndarray'] = ancillary_dataset_array
    ancillary_dataset_dict['transform'] = ancillary_dataset_transform
    ancillary_dataset_dict['projection'] = ancillary_dataset_projection
    return ancillary_dataset_dict


def write_product(outType, outDir, outName, array, projection, transform,
                  geoTiff=True) -> None:
    try:
        cols = array.shape[0]
        rows = array.shape[1] if len(
            array.shape) > 1 else 1
        fileType = '.tif' if geoTiff else '.bin'
        imageName = os.path.join(outDir, outName + fileType)
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
        print('Wrote annual {} products to: {}'.format(outType, imageName))
    except Exception as e:
        msg = f'Encountered {str(e)} during write out using GDAL'
        raise RuntimeError(msg)
    return None

def search_ancillary_path(ancillary_base_path: str, tile: str) -> str:
    if not os.path.exists(ancillary_base_path):
        msg = f'HALT! {ancillary_base_path} does not exist'
        raise FileNotFoundError(msg)
    ancillary_file_name = f'{ANCILLARY_PRE_STR}{tile}{ANCILLARY_POST_STR}'
    ancillary_file_path = os.path.join(ancillary_base_path,
                                       ancillary_file_name)
    logging.info(f'Looking for {ancillary_file_path}')
    if os.path.exists(ancillary_file_path):
        return ancillary_file_path
    else:
        msg = f'{ancillary_file_path} not found'
        raise FileNotFoundError(msg)


def read_ancillary_path(ancillary_file_path: str) -> dict:
    try:
        logging.info(f'Attempting to read {ancillary_file_path}')
        ancillary_dataset_object = gdal.Open(ancillary_file_path)
        ancillary_dataset_band = ancillary_dataset_object.GetRasterBand(1)
        ancillary_dataset_array = ancillary_dataset_band.ReadAsArray()
        ancillary_dataset_transform = \
            ancillary_dataset_object.GetGeoTransform()
        ancillary_dataset_projection = \
            ancillary_dataset_object.GetProjection()
    except Exception as e:
        msg = 'Encountered while trying' + \
            f' to read {ancillary_file_path} with GDAL'
        raise RuntimeError(str(e) + msg)
    ancillary_dataset_dict = {}
    ancillary_dataset_dict['ndarray'] = ancillary_dataset_array
    ancillary_dataset_dict['transform'] = ancillary_dataset_transform
    ancillary_dataset_dict['projection'] = ancillary_dataset_projection
    return ancillary_dataset_dict

def main() -> None:
    desc = 'Use this application to apply ' + \
        'ancillary masks to MODAPS test products.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-y',
                        '--year',
                        type=int,
                        required=True,
                        dest='year',
                        help='year to apply')

    parser.add_argument('-t',
                        '--tile',
                        type=str,
                        required=True,
                        dest='tile',
                        help='tile to apply')

    parser.add_argument('-hd',
                        '--hdf-dir',
                        type=str,
                        required=True,
                        dest='hdf_dir',
                        help='Directory to HDF products')

    parser.add_argument('-pd',
                        '--prd-code',
                        type=str,
                        required=True,
                        dest='prod_code',
                        help='Prod code')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        required=False,
                        dest='output_dir',
                        help='Path to output directory')

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

    return diff_products(args.year,
                         args.tile,
                         args.hdf_dir,
                         args.output_dir,
                         args.prod_code)


if __name__ == '__main__':
    sys.exit(main())
