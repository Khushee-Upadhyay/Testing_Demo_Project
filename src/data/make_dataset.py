# -*- coding: utf-8 -*-
import logging
import pandas as pd
import os


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(input_filepath)
    # creating new feature by using age column

    df["age_range"] = 1000
    for i in range(len(df["age"])):
        if df["age"][i] < 30:
            df["age_range"][i] = 1
        elif 30 <= df["age"][i] < 45:
            df["age_range"][i] = 2
        elif df["age"][i] >= 45:
            df["age_range"][i] = 3

    df["have_children"] = ["No" if i == 0 else "Yes" for i in df["children"]]
    df.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    print(os.getcwd())
    input_path = "data/raw/insurance.csv"
    output_path = "data/processed/processed_insurance_data.csv"
    main(input_path, output_path)
