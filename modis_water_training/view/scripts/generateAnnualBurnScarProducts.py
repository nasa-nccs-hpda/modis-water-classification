#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import os
import numpy as np
from glob import glob
from pprint import pprint
from osgeo import gdal
import matplotlib.pyplot as plt


def main():

    desc = 'This application runs Test_Burned_Area_Alpha_0.0.1'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--multiyear',
                        action='store_true',
                        help='More than one year to run.')

    parser.add_argument('-years',
                        type=str,
                        required='--multiyear' in sys.argv)

    parser.add_argument('-year',
                        type=int,
                        required='--multiyear' not in sys.argv,
                        help='Year to generate annual burn scar products.')

    parser.add_argument('-tile',
                        type=str,
                        required=True,
                        help='Tile to use for burn scar products.')

    args = parser.parse_args()

    year = args.year

    tile = '*{}*.hdf'.format(args.tile)

    path_ = '/css/modis/Collection6/L3/MCD64A1-BurnArea/'

    output_path = '/att/nobackup/cssprad1/projects/' +\
        'modis_water/data/burn_scar_products'

    if args.multiyear:
        years = [int(y) for y in args.years.split(' ')]
        for y in years:
            print('Processing year: {}, tile: {}'.format(y, tile))
            generateProduct(year=y, tile=tile, path=path_,
                            output_path=output_path)
    else:
        generateProduct(year=year, tile=tile, path=path_,
                        output_path=output_path)


def generateProduct(year, tile, path, output_path):
    subdirhdfs = getAllFiles(path=path, year=year, tile=tile)

    matSet = [getMatFromHDF(subdir, 'Burn Date', 'Uncertainty')
              for subdir in subdirhdfs]

    outmat = logical_or_mat(matSet)

    x, y = np.histogram(outmat)

    print(x)
    print(y)

    outpath = setupOutput(year=year, tile=tile, output_path=output_path)

    geo, proj, ncols, nrows = getRasterInfo(subdirhdfs[0])

    output_raster(outPath=outpath, outmat=outmat, geo=geo,
                  proj=proj, ncols=ncols, nrows=nrows)


def getAllFiles(path, year, tile):
    path_to_prepend = os.path.join(path, str(year))
    subdirs = sorted(os.listdir(path_to_prepend))
    subdirs = [os.path.join(path_to_prepend, subdir) for subdir in subdirs]
    subdirhdfs = [glob(os.path.join(subdir, tile))[0] for subdir in subdirs]
    pprint(subdirhdfs)
    return subdirhdfs


def getMatFromHDF(hdf, substr, excludeStr):
    hdf = gdal.Open(hdf)
    subd = [sd for sd, _ in hdf.GetSubDatasets(
    ) if substr in sd and excludeStr not in sd][0]
    print('Opening: {}'.format(subd))
    del hdf
    ds = gdal.Open(subd)
    mat = ds.GetRasterBand(1).ReadAsArray()
    mat = np.where(mat > 0, 1, 0)
    del ds
    return mat


def logical_or_mat(mat_list):
    output_mat = np.empty(mat_list[0].shape)
    for mat in mat_list:
        output_mat = output_mat + mat
        mat = None
    output_mat = np.where(output_mat > 0, 1, 0)
    plt.figure(figsize=(15, 15))
    plt.matshow(output_mat, fignum=1)
    return output_mat


def setupOutput(year, tile, output_path):
    file_name = 'MCD64A1-BurnArea_Annual_A{}.{}.tif'.format(
        year, tile.replace('*', '').replace('.hdf', ''))
    dir_output = 'MCD64A1-BurnArea-Annual/{}'.format(year)
    output_dir_full = os.path.join(output_path, dir_output)
    print(output_dir_full)
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)
    outPath = os.path.join(output_dir_full, file_name)
    print(outPath)
    return outPath


def getRasterInfo(file):
    ds = gdal.Open(file, gdal.GA_ReadOnly)
    subd = [sd for sd, _ in ds.GetSubDatasets(
    ) if 'Burn Date' in sd and 'Uncertainty' not in sd][0]
    ds = gdal.Open(subd, gdal.GA_ReadOnly)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    print('Transform')
    print(geo)
    print('Projection')
    print(proj)
    print('Width')
    print(ncols)
    print('Height')
    print(nrows)
    ds = None
    return geo, proj, ncols, nrows


def output_raster(outPath, outmat, geo, proj, ncols, nrows):
    # Output predicted binary raster masked with good-bad mask.
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(outPath, ncols, nrows, 1,
                          gdal.GDT_Int16, options=['COMPRESS=LZW'])
    outDs.SetGeoTransform(geo)
    outDs.SetProjection(proj)
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(outmat)
    outDs.FlushCache()
    outDs = None
    outBand = None
    driver = None


if __name__ == '__main__':
    sys.exit(main())
