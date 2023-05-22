import os
import glob
import sys

import dask.bag as db
from dask.diagnostics import ProgressBar

from osgeo import gdal


MODAPS_C61_BASEPATH: str = '/explore/nobackup/projects/ilab/data/MODIS/' + \
    'PRODUCTION/MODAPS_test2_05112023/MOD44W/2019/001'


def get_output_name_given_input(input_subdataset_path: str, 
                                outdir: str = '.') -> str:
    subdataset_name = input_subdataset_path.split('"')
    basename = os.path.basename(subdataset_name[1])
    bandname = subdataset_name[-1].split(':')[-1]
    outname = basename.replace('.hdf', f'.{bandname}.tif')
    outpath = os.path.join(outdir, outname)
    return outpath


def get_subdatasets(hdf_path: str) -> list:
    hdf_dataset = gdal.Open(hdf_path, gdal.GA_ReadOnly)
    subdatasets = hdf_dataset.GetSubDatasets()
    hdf_dataset = None
    return subdatasets

def hdf_to_geotiff(hdf_path: str, outdir: str) -> int:
    subdatasets = get_subdatasets(hdf_path)
    for subdatasetpath, subdataset_description in subdatasets:
        subdataset_output_path = get_output_name_given_input(
            subdatasetpath, outdir=outdir)
        gdal.Translate(subdataset_output_path, 
                       subdatasetpath, 
                       format="GTiff", 
                       options=['COMPRESS=LZW'])
    return 1


def main() -> None:
    outdir = '/explore/nobackup/projects/ilab/data/MODIS/PRODUCTION/' + \
        'MODAPS_test2_05112023_geotiff/MOD44W/2019/001'
    hdf_post_str = '*.hdf'
    
    hdf_regex = os.path.join(MODAPS_C61_BASEPATH, hdf_post_str)
    all_modaps_hdfs = sorted(glob.glob(hdf_regex))
    print(f'>> Found {len(all_modaps_hdfs)} HDFS to process')
    if len(all_modaps_hdfs) < 1:
        raise FileNotFoundError(f'Could not find HDFs matching {hdf_regex}')
    hdf_paths_bag = db.from_sequence(all_modaps_hdfs)
    with ProgressBar():
        db.map(hdf_to_geotiff, hdf_paths_bag, outdir=outdir).compute()


if __name__ == '__main__':
    sys.exit(main())