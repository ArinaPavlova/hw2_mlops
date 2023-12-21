import app.main
from minio import Minio


def test_create_minio(mocker):
    mocker.patch('app.main.create_minio', return_value={})
    assert app.main.create_minio() == {}

def test_model_list(mocker):
    mocker.patch('app.main.create_minio', return_value={})
    assert "LogisticRegression" in str(app.main.Models().available_model_list('classification'))

def test_delete_model(mocker):
    mocker.patch('app.main.create_minio', return_value={1: "bla-bla"})
    mocker.patch('app.main.del_model_from_minio', return_value={})
    assert app.main.Models().delete_model(1) == 200
