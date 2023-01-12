import os
import sys
import glob
import pandas as pd


def main():
    outputPath = './MOD09_model_training_data_special_added.csv'
    mainTrainingDataPath = '/att/nobackup/cssprad1/projects/modis_water/data/training_data/v1.1.0/example.csv'
    additionalTrainingDataDir = '/att/nobackup/cssprad1/projects/modis_water/data/training_data/v2.0.0/'

    additionalTrainingCSVs = [fv for fv in sorted(
        glob.glob(os.path.join(additionalTrainingDataDir, '*.csv')))]
    additionalTrainingDFs = [pd.read_csv(fv) for fv in additionalTrainingCSVs]
    totalTrainingDFs = [pd.read_csv(mainTrainingDataPath)].extend(
        additionalTrainingDFs)
    
    totalDF = pd.concat(totalTrainingDFs)
    totalDF = totalDF.sample(frac=1)
    totalDF = totalDF.reset_index()
    totalDF = totalDF.drop(['index'], axis=1)
    
    print('Writing {} rows to {}'.format(len(totalDF.index), outputPath))
    totalDF.to_csv(outputPath)


if __name__ == '__main__':
    sys.exit(main())
