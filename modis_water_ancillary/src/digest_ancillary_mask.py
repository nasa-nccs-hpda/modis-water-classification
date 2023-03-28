import os
import logging

import numpy as np
from osgeo import gdal

from ingest_mw_hdf import ingest_mw_hdf
from modis_water.model.SevenClass import SevenClassMap

"""
QA LEGEND

Legend=
    1: High Confidence Observation
    2: Low Confidence Water, but MOD44W C5 is water
    3: Low Confidence Land
    4: Ocean Mask
    5: Ocean Mask but no water detected
    6: Burn Scar (from MCD64A1)
    7: Urban/Impervious surface
    8: No water detected, Collection 5 shows water
    10: No data (outside of projected area)

SEVEN CLASS LEGEND

Legend=
    0: shallow ocean
    1: Land
    2: Shoreline
    3: Inland water
    4: Deep inland water
    5: Ephemeral water
    6: Moderate ocean
    7: Deep ocean

ANCILLARY MASK LEGEND

Legend=
    0: not land
    1: definitely land
    2: ocean
    10: outside projection
"""
ANCILLARY_WATER: int = 0
ANCILLARY_LAND: int = 1
ANCILLARY_OCEAN: int = 2
ANCILLARY_NODATA: int = 10

MW_WATER_MASK: str = 'water_mask'
MW_QA_MASK: str = 'qa_mask'
SEVEN_CLASS_MASK: str = 'seven_class'

ANCILLARY_PRE_STR: str = 'Dyn_Water_Ancillary_'
ANCILLARY_POST_STR: str = '_v3b.tif'


def update_mask_products(year: int,
                         tile: str,
                         hdf_base_path: str,
                         ancillary_base_path: str,
                         static_seven_class_dir: str) -> dict:
    mw_data_dict = ingest_mw_hdf(hdf_base_path, year, tile)
    ancillary_file_path = search_ancillary_path(ancillary_base_path, tile)
    ancillary_data_dict = read_ancillary_path(ancillary_file_path)
    ancillary_array = ancillary_data_dict['ndarray']
    modified_mw_data_array = copy_and_modify_water(
        mw_data_dict[MW_WATER_MASK], ancillary_array)
    modified_qa_data_array = copy_and_modify_qa(
        mw_data_dict[MW_QA_MASK],
        mw_data_dict[MW_WATER_MASK],
        ancillary_array
    )
    modified_seven_class_data_array = copy_and_modify_seven_class(
        modified_mw_data_array,
        tile,
        static_seven_class_dir,
        ancillary_array
    )
    modified_data_dict = {}
    modified_data_dict[MW_WATER_MASK] = modified_mw_data_array
    modified_data_dict[MW_QA_MASK] = modified_qa_data_array
    modified_data_dict[SEVEN_CLASS_MASK] = modified_seven_class_data_array
    modified_data_dict['projection'] = ancillary_data_dict['projection']
    modified_data_dict['transform'] = ancillary_data_dict['transform']
    return modified_data_dict


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


def copy_and_modify_water(water_dict: dict,
                          ancillary_array: np.ndarray) -> np.ndarray:
    logging.info('Modifying water mask')
    water_mask_array = water_dict['ndarray']
    modified_water_mask_array = np.where(
        ancillary_array == ANCILLARY_OCEAN, 1, water_mask_array)
    modified_water_mask_array = np.where(
        ancillary_array == ANCILLARY_LAND, 0, modified_water_mask_array)
    modified_water_mask_array = np.where(
        ancillary_array == ANCILLARY_NODATA, 255, modified_water_mask_array)
    return modified_water_mask_array


def copy_and_modify_qa(qa_dict: dict, water_dict: dict,
                       ancillary_array: np.ndarray) -> np.ndarray:
    logging.info('Modifying QA mask')
    qa_array = qa_dict['ndarray']
    water_array = water_dict['ndarray']
    qa_array_modified = np.where(qa_array == 4, 9, qa_array)
    ocean_mask_no_water = (
        (ancillary_array == ANCILLARY_OCEAN) & (water_array == 0))
    ocean_mask_water = ((ancillary_array == ANCILLARY_OCEAN)
                        & (water_array == 1))
    land_pixel_switch = ((ancillary_array == ANCILLARY_LAND)
                         & (water_array == 1))
    no_data = (ancillary_array == ANCILLARY_NODATA)
    qa_array_modified = np.where(ocean_mask_no_water, 5, qa_array_modified)
    qa_array_modified = np.where(ocean_mask_water, 4, qa_array_modified)
    qa_array_modified = np.where(no_data, 10, qa_array_modified)
    qa_array_modified = np.where(land_pixel_switch, 20, qa_array_modified)
    return qa_array_modified


def copy_and_modify_seven_class(modified_water_array: dict,
                                tile: str,
                                static_seven_class_dir: str,
                                ancillary_array: np.ndarray) -> np.ndarray:
    logging.info('Modifying seven class')
    modified_seven_class_array = SevenClassMap.generateSevenClass(
        modified_water_array, tile, static_seven_class_dir)
    no_data = (ancillary_array == ANCILLARY_NODATA)
    modified_seven_class_array = np.where(
        no_data, 255, modified_seven_class_array)
    return modified_seven_class_array
