import argparse
import sys

# sys.path.append('.')
from modis_water_training.model.TabularModisDataGeneratorMasked import TabularModisDataGeneratorMasked
from modis_water_training.model.TabularModisDataGenerator import TabularModisDataGenerator

# -----------------------------------------------------------------------------
# main
#
# modis_water_random_forest.view.TabularModisDataGeneratorView.py -d 167 170 -i /css/modis/Collection6.1/L2G/MOD09GA -o . -m /att/nobackup/cssprad1/projects/modis_water/data/water_masks/Buffered2000_2019 -t 17 02 -y 2020
# -----------------------------------------------------------------------------


def main(client=None):

    desc = 'Use this application to generate MODIS Water detection' + \
        ' training data.'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-d',
                        nargs="+",
                        help='range of julian day of year; \n' +
                        'in individual day form: 167 170. Upper bound exclusive.')

    parser.add_argument('-e',
                        nargs="+",
                        help='julian day of year to exclude; \n' +
                        'in individual day form: 169 170')

    parser.add_argument('-i',
                        default='.',
                        help='Path to input directory')

    parser.add_argument('-o',
                        default='.',
                        help='Path to output directory')

    parser.add_argument('--special_mask',
                        action='store_true',
                        help='If using a special mask instead of water mask')

    parser.add_argument('-sp',
                        required=True if '--special_mask' in sys.argv
                        else False,
                        help='Path for special mask. ' +
                        'Required if using --special_mask')

    parser.add_argument('--noQA',
                        required=False,
                        action='store_true',
                        help='Do not apply QA mask')

    parser.add_argument('-m',
                        default='.',
                        required=False if '--special_mask' in sys.argv
                        else True,
                        help='Path to water mask directory.' +
                        ' Not required if using --special_mask.')

    parser.add_argument('--water',
                        action='store_true',
                        help='If water vals or land vals.' +
                        ' Required if using --special_mask.')

    parser.add_argument('-t',
                        nargs='+',
                        help='Tile to process; 9 5')

    parser.add_argument('-y',
                        type=int,
                        help='Year to use.')

    parser.add_argument('-n',
                        type=int,
                        default=100000,
                        help='Number of data points to generate.')

    parser.add_argument('--rm',
                        action='store_true',
                        help='Delete individual days CSV files.')

    args = parser.parse_args()

    if args.d:
        julianDays = [day for day in map(int, args.d)]
        if len(julianDays) % 2 != 0:
            raise RuntimeError('Give even number of ranges')
        it = iter(julianDays)
        julianRangeTotal = [range(day[0], day[1]) for day in zip(it, it)]
        julianDays = []
        for r in julianRangeTotal:
            julianDays.extend([*r])

    else:
        julianDays = None

    tile = tuple(map(int, args.t))
    tile = 'h{:02}v{:02}'.format(tile[0], tile[1])

    if args.e:
        daysToExclude = list(map(int, args.e))
    else:
        daysToExclude = None

    if args.special_mask:
        generator = TabularModisDataGeneratorMasked(
            tile=tile,
            year=args.y,
            maskPath=args.sp,
            water=args.water,
            tileDir=args.i,
            outDir=args.o,
            julianDays=julianDays,
            numDataPoints=args.n,
            excludeDays=daysToExclude,
            rm_csv=args.rm,
            noQA=args.noQA,
        )

    else:

        generator = TabularModisDataGenerator(
            tile,
            args.y,
            tileDir=args.i,
            outDir=args.o,
            waterMaskDir=args.m,
            julianDays=julianDays,
            numDataPoints=args.n,
            excludeDays=daysToExclude,
            rm_csv=args.rm)

    generator.run()


if __name__ == '__main__':
    sys.exit(main())
