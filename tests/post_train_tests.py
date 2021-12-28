import numpy as np


def test_sex_invariance(model):
    female_sample = [19, 1, 27.9, 0, 1, 2, 1, 1]
    male_sample = [19, 0, 27.9, 0, 1, 2, 1, 1]
    result_female_sample = model.predict(np.array(female_sample).reshape(1, -1))
    result_male_sample = model.predict(np.array(male_sample).reshape(1, -1))
    assert result_female_sample == result_male_sample

