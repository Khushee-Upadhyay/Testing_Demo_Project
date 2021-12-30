Pre-train and Post-train tests for insurance prediction problem
==============================

Demo project for writing test cases for Machine Learning problems.


Dataset: "<a target="_blank" href="https://www.kaggle.com/mirichoi0218/insurance">Medical Cost Personal Datasets</a>



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README file
    ├── data
    │   ├── model_testing_data       <- Data for validation
    │   ├── model_training_data        <- Data for training models
    │   ├── processed      <- The final processed data set
    │   └── raw            <- The original, immutable data dump.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data          
    │   │   └── make_dataset.py  <- Scripts to generate pre-processed data
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |
    ├── tests
    |   |
    |   │   └── pre_train_tests.py  <- Pre-train test script
    |   │   └── post_train_tests.py  <- Post-train test script


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
