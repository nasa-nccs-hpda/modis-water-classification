import glob
import os
import logging

from osgeo import gdal

HDF_NAME_PRE_STR: str = 'MOD44W.A'
HDF_PRE_STR: str = 'HDF4_EOS:EOS_GRID:"'
WATER_MASK_POST_STR: str = '":MOD44W_250m_GRID:water_mask'
SEVEN_CLASS_POST_STR: str = '":MOD44W_250m_GRID:seven_class'
QA_MASK_POST_STR: str = '":MOD44W_250m_GRID:water_mask_QA'
MASK_POST_STR_LIST: list = [WATER_MASK_POST_STR,
                            SEVEN_CLASS_POST_STR, QA_MASK_POST_STR]
RASTER_BAND_NUM: int = 1

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
"""


def ingest_mw_hdf(hdf_base_path: str, year: int, tile: str) -> dict:
    hdf_file_path = search_hdf_file_path(hdf_base_path, year, tile)
    water_mask_path = get_hdf_subdataset_path(hdf_file_path,
                                              WATER_MASK_POST_STR)
    qa_mask_path = get_hdf_subdataset_path(hdf_file_path,
                                           QA_MASK_POST_STR)
    seven_class_path = get_hdf_subdataset_path(hdf_file_path,
                                               SEVEN_CLASS_POST_STR)
    water_mask_dict = read_hdf(water_mask_path)
    qa_mask_dict = read_hdf(qa_mask_path)
    seven_class_dict = read_hdf(seven_class_path)
    mask_dict_to_return = {}
    mask_dict_to_return['water_mask'] = water_mask_dict
    mask_dict_to_return['qa_mask'] = qa_mask_dict
    mask_dict_to_return['seven_class'] = seven_class_dict
    return mask_dict_to_return


def search_hdf_file_path(hdf_base_path: str, year: int, tile: str) -> str:
    if not os.path.exists(hdf_base_path):
        msg = f'HALT! {hdf_base_path} does not exist'
        raise FileNotFoundError(msg)
    hdf_regex = f'{HDF_NAME_PRE_STR}{year}001.{tile}.061.*.hdf'
    hdf_full_path_regex = os.path.join(hdf_base_path, hdf_regex)
    logging.info(f'Searching for {hdf_full_path_regex}')
    hdf_glob = sorted(glob.glob(hdf_full_path_regex))
    hdf_glob_len = len(hdf_glob)
    if hdf_glob_len < 1:
        msg = f'{hdf_glob} not files matching this regex'
        raise FileNotFoundError(msg)
    elif hdf_glob_len > 1:
        msg = f'Found {hdf_glob_len} files ' + \
            f'matching this regex {hdf_full_path_regex}'
        raise RuntimeError(msg)
    else:
        logging.info(f'Found file: {hdf_glob[0]}')
        return hdf_glob[0]


def get_hdf_subdataset_path(hdf_file_path: str,
                            subdataset_post_str: str) -> str:
    return f'{HDF_PRE_STR}{hdf_file_path}{subdataset_post_str}'


def read_hdf(hdf_file_path: str) -> dict:
    try:
        logging.info(f'Attempting to open {hdf_file_path}')
        hdf_dataset_object = gdal.Open(hdf_file_path)
        hdf_dataset_band = hdf_dataset_object.GetRasterBand(1)
        hdf_dataset_array = hdf_dataset_band.ReadAsArray()
    except Exception as e:
        msg = f'Encountered while trying to read {hdf_file_path} with GDAL'
        raise RuntimeError(str(e) + msg)
    dataset_dict = {}
    dataset_dict['ndarray'] = hdf_dataset_array
    logging.info(f'Extracted {hdf_file_path}')
    return dataset_dict
