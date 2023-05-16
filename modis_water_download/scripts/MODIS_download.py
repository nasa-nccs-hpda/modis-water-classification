### Downloads LAADS DAAC data with parallel processing, specifically MODIS daily products
### Can be used to repair incomplete downloads
### Currently handles one year and one tile, upgrade to handle csv of inputs

import os
import glob
import sys
import csv
import time
import argparse
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, Lock, cpu_count

def getParser():
    """
    Get parser object for main initialization.
    """
    desc = 'Use this to specify the tile and year to download MODIS data from LAADS DAAC. ' + \
        'Regions follow the MODIS tile system.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-t', '--tile', type=str, required=True,
        help='Tile designator for requested download')
    
    parser.add_argument(
        '-a', '--authorization', type=str, required=True,
        help='Download token for your LAADS DAAC account')
    
    parser.add_argument(
        '-c', '--collection', type=str, required=True,
        help='Collection of MODIS data (no decimal points, i.e. 6.1 = 61)')
    
    parser.add_argument(
        '-o', '--outPath', type=str, default='.',
        help='Output directory (Parent collection i.e. L3/L2G)')
    
    parser.add_argument(
        '-dp', '--dataProduct', type=str, required=True,
        help='Data product requested')

    parser.add_argument(
        '-y', '--year', type=int, required=True,
        help='Year of requested data product download')
    
    parser.add_argument(
        '-l', '--tiles_from_list', type=str, required=False,
        help='Optional pass in a directory of csvs to query tiles (for VIIRS)')
    

    return parser.parse_args()

def download_file(download_url: str):
    os.system(download_url)
    return

def main():

    # Process command-line args.
    args = getParser()

    #Arguments
    tile = args.tile
    auth = args.authorization
    collection = args.collection
    outPath = args.outPath
    dp = args.dataProduct
    year = args.year

    #dir for csvs argument
    tarPath = args.tiles_from_list

    # Set logging
    logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
    timer = time.time()
    logging.info(f'Downloading {dp} for year: {year} and tile: {tile}')

    #if basic tile call
    if args.tiles_from_list is None:
        dataIDs = []
        if year%4 == 0: #Check for leap year
            nDays = 366
        else:
            nDays = 365
        days = np.linspace(1,nDays, num=nDays, endpoint=True, dtype=int)
        days = [format(x, '03d') for x in days]
        tile_list = [dp+'.'+f'A{year}{day}.{tile}*.hdf' for day in days]
        logging.info(f'Expecting {nDays} files')

        #initialize list of downloads and functions
        download_urls = []

        #Define output path, build if it doesn't exist
        output = os.path.join(outPath,dp,str(year))
        if not os.path.isdir(output):
            os.makedirs(output,exist_ok=True)

        #Distribute the days, not the filename. wget will automatically avoid duplicates
        for day in days:
            download_command = f'wget -e robots=off -m -np -A "*{args.tile}*" ' + \
                f'-nd "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{args.collection}/{args.dataProduct}/{args.year}/{day}/" ' + \
                f'--header "Authorization: Bearer {args.authorization}" -P {output}'
            download_urls.append(download_command)
        
        logging.info(f'Downloading {len(download_urls)} tiles.')
    #if calling tiles from list of csvs
    else:
        #create list of tuples
        targets = []
        for filename in os.listdir(tarPath):
            #VIIRS only goes for > 2011
            if int(os.path.splitext(filename)[0][-4:]) > 2011:
                print(filename)
                with open(os.path.join(tarPath,filename)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    next(csv_reader)
                    for row in csv_reader:
                        year_date_ID= [row[3][:-2],row[4][:-2].zfill(3),row[5]]
                        targets.append(year_date_ID)
                        print(year_date_ID)
        logging.info(f'Expecting {len(targets)} files')

        #initialize list of downloads
        download_urls = []

        for combo in targets:
            #define outputs
            output = os.path.join(outPath,dp,str(combo[0]))
            if not os.path.isdir(output):
                os.makedirs(output,exist_ok=True)            
            download_command = f'wget -e robots=off -m -np -A "*{combo[2]}*" ' + \
                f'-nd "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{args.collection}/{args.dataProduct}/{combo[0]}/{combo[1]}/" ' + \
                f'--header "Authorization: Bearer {args.authorization}" -P {output}'
            download_urls.append(download_command)

    
    # Set pool, start parallel multiprocessing
    p = Pool(processes=20)
    p.map(download_file, download_urls)
    p.close()
    p.join()
    
    #Start logging the job
    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min, output at {output}.')

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
