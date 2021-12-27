# -*- coding: utf-8 -*-
import logging
import os
from sklearn.preprocessing import LabelEncoder

from src.util.csv_data_operations import load_insurance_data, save_insurance_data


def data_preprocessing():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # df = pd.DataFrame()
    df = load_insurance_data()
    # creating new feature by using age column

    df["age_range"] = 1000
    for i in range(len(df["age"])):
        if df["age"][i] < 30:
            df["age_range"][i] = 1
        elif 30 <= df["age"][i] < 40:
            df["age_range"][i] = 2
        elif 40 <= df["age"][i] < 50:
            df["age_range"][i] = 3
        elif df["age_range"][i] >= 50:
            df["age_range"][i] = 4

    df["have_children"] = ["No" if i == 0 else "Yes" for i in df["children"]]

    cat_variable = ['sex', 'smoker', 'region', 'have_children']
    lb = LabelEncoder()
    df[cat_variable] = df[cat_variable].apply(lambda col: lb.fit_transform(col.astype(str)))

    save_insurance_data(df)
    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    print(os.getcwd())

    data_preprocessing()
