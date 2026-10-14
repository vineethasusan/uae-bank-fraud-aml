import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def bank():
    from fincrime import data

    return data.generate(n_customers=800, seed=11)
