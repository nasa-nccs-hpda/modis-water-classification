import os
import glob
import sys
# import logging
# asfrom osgeo import gdal


GA: str = 'MOD09GA'
GQ: str = 'MOD09GQ'
MCD: str = 'MCD64A1'
PRODUCT_TYPES: list = [GA, GQ, MCD]
DATA_DIR: str = 'data'
FILE_TYPE: str = 'hdf'


def main():
    tile = sys.argv[1]
    year = sys.argv[2]
    print(f'Tile: {tile}, Year: {year}')
    file_dict = {}
    for product in PRODUCT_TYPES:
        directory = os.path.join(year, tile, DATA_DIR, product, year)
        file_regex = os.path.join(directory, f'*.{FILE_TYPE}')
        file_dict[product] = {}
        file_dict[product]['files'] = sorted(glob.glob(file_regex))
        file_dict[product]['days'] = [os.path.basename(prd_file).split('.')[1]
                                      for prd_file in file_dict[product]['files']]

    mod09ga_days = set(file_dict[GA]['days'])
    mod09gq_days = set(file_dict[GQ]['days'])
    print(f'Total GA days: {len(mod09ga_days)}')
    print(f'Total GQ days: {len(mod09gq_days)}')
    nodiff = mod09ga_days & mod09gq_days
    mod09ga_diff = [day for day in mod09ga_days if day not in nodiff]
    mod09gq_diff = [day for day in mod09gq_days if day not in nodiff]
    print(f'Found that needs a GQ pair {len(mod09ga_diff)}')
    print(f'Found that needs a GA pair {len(mod09gq_diff)}')
    print(mod09ga_diff)
    print(mod09gq_diff)
    mod09gq_days_formatted = [mod09gq_diff_day[5:]+'\n' for mod09gq_diff_day in mod09gq_diff]
    mod09ga_days_formatted = [mod09ga_diff_day[5:]+'\n' for mod09ga_diff_day in mod09ga_diff]
    with open(f'{tile}.{year}.gq', 'w') as f:
        f.writelines(mod09gq_days_formatted)
    with open(f'{tile}.{year}.ga', 'w') as f:
        f.writelines(mod09ga_days_formatted)


if __name__ == '__main__':
    sys.exit(main())

