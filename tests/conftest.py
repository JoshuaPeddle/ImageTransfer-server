import sys
import pytest
sys.path.append("..") # Adds higher directory to python modules path.
from ..app import app as flask_app  # noqa: E402
@pytest.fixture()
def app():
    
    # other setup can go here

    yield flask_app

    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()

