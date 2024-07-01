import pytest
from sklearn.datasets import load_breast_cancer

from ml_utils import config

@pytest.fixture(scope='module')
def breast_cancer_data():
    yield load_breast_cancer(return_X_y=True, as_frame=True)

def test_auc(breast_cancer_data):
    X, y = breast_cancer_data
    assert config['input']['id_col'] is None
