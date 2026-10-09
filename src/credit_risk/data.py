"""Loading, validation and cleaning of the UCI 'default of credit card clients' data."""
from __future__ import annotations

import hashlib
import urllib.request

import pandas as pd

from . import config as C


def download(force: bool = False) -> None:
    """Fetch the raw .xls and verify its checksum."""
    C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if C.RAW_PATH.exists() and not force:
        return
    urllib.request.urlretrieve(C.DATA_URL, C.RAW_PATH)
    digest = hashlib.sha256(C.RAW_PATH.read_bytes()).hexdigest()
    if digest != C.DATA_SHA256:
        raise ValueError(f"Checksum mismatch for {C.RAW_PATH}: {digest}")


def load_raw() -> pd.DataFrame:
    download()
    return pd.read_excel(C.RAW_PATH, header=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known coding issues in this dataset.

    * PAY_0 is renamed to PAY_1 so the six status columns line up with BILL/PAY_AMT 1..6.
    * EDUCATION has undocumented codes 0, 5, 6 -> merged into 4 ("others").
    * MARRIAGE has an undocumented code 0 -> merged into 3 ("others").
    """
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": C.TARGET})
    df = df.drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Cheap schema checks that fail loudly instead of training on bad data."""
    assert df[C.TARGET].isin([0, 1]).all(), "target must be binary"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    assert (df["LIMIT_BAL"] > 0).all(), "credit limit must be positive"
    assert df["EDUCATION"].isin([1, 2, 3, 4]).all()
    assert df["MARRIAGE"].isin([1, 2, 3]).all()
    assert df[C.PAY_STATUS].isin(range(-2, 10)).all().all()


def load() -> pd.DataFrame:
    return clean(load_raw())
