import argparse
import glob
import logging
import os
import sys
import tqdm

from digest_ancillary_mask import update_mask_products
from regurgitate_mw_hdf import write_out


MW_WATER_MASK: str = 'water_mask'
MW_QA_MASK: str = 'qa_mask'
SEVEN_CLASS_MASK: str = 'seven_class'
LAYER_TYPE_LIST: list = [MW_WATER_MASK, MW_QA_MASK, SEVEN_CLASS_MASK]


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def ancillary_pipeline(year: int,
                       tile: str,
                       hdf_base_path: str,
                       ancillary_base_path: str,
                       static_seven_class_dir: str,
                       out_dir: str) -> int:
    modified_products = update_mask_products(
        year, tile, hdf_base_path, ancillary_base_path, static_seven_class_dir)
    projection = modified_products['projection']
    transform = modified_products['transform']
    for product_type in LAYER_TYPE_LIST:
        print(f'Processing {product_type} for tile {tile} for year {year}')
        array = modified_products[product_type]
        write_out(array, projection, transform,
                  out_dir, product_type, year, tile)
    return 0


def main_run_all() -> None:
    desc = 'Use this application to apply ' + \
        'ancillary masks to MODAPS test products.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-y',
                        '--year',
                        type=int,
                        required=True,
                        dest='year',
                        help='year to apply')

    parser.add_argument('-a',
                        '--ancillary-dir',
                        type=str,
                        required=True,
                        dest='ancillary_dir',
                        help='Path to ancillary dir')

    parser.add_argument('-hd',
                        '--hdf-dir',
                        type=str,
                        required=True,
                        dest='hdf_dir',
                        help='Directory to HDF products')

    parser.add_argument('-sc',
                        '--seven-class',
                        type=str,
                        required=True,
                        dest='seven_class',
                        help='Directory to static seven class')

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
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{args.year}.all.ancillary.mod.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    hdfs_regex = os.path.join(args.hdf_dir, '*.hdf')
    hdfs_to_process = sorted(glob.glob(hdfs_regex))
    if len(hdfs_to_process) < 1:
        msg = f'Could not find any hdfs at {hdfs_regex}'
        raise FileNotFoundError(msg)
    logging.info(f'Found {len(hdfs_to_process)} to process')
    tiles = [os.path.basename(hdf_file_path).split('.')[2]
             for hdf_file_path in hdfs_to_process]

    logging.info(f'Processing {len(tiles)} tiles')
    for tile in tqdm.tqdm(tiles):
        try:
            _ = ancillary_pipeline(args.year,
                                   tile,
                                   args.hdf_dir,
                                   args.ancillary_dir,
                                   args.seven_class,
                                   args.output_dir)
        except Exception as e:
            msg = f'Encountered exception {str(e)}'
            logging.info(msg)
            continue
    return 0


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

    parser.add_argument('-a',
                        '--ancillary-dir',
                        type=str,
                        required=True,
                        dest='ancillary_dir',
                        help='Path to ancillary dir')

    parser.add_argument('-hd',
                        '--hdf-dir',
                        type=str,
                        required=True,
                        dest='hdf_dir',
                        help='Directory to HDF products')

    parser.add_argument('-sc',
                        '--seven-class',
                        type=str,
                        required=True,
                        dest='seven_class',
                        help='Directory to static seven class')

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
    fh = logging.FileHandler(f'{args.year}.{args.tile}.ancillary.mod.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return ancillary_pipeline(args.year,
                              args.tile,
                              args.hdf_dir,
                              args.ancillary_dir,
                              args.seven_class,
                              args.output_dir)


if __name__ == '__main__':
    sys.exit(main_run_all())
