"""FastAPI scoring service.

    uvicorn credit_risk.serve:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/request.json
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from . import config as C
from .features import RAW_FEATURES, engineer

app = FastAPI(title="Credit Default Risk API", version="1.0.0")


class Applicant(BaseModel):
    LIMIT_BAL: float = Field(gt=0, description="Credit limit")
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY: list[int] = Field(min_length=6, max_length=6, description="Repayment status, most recent month first (-2..8)")
    BILL_AMT: list[float] = Field(min_length=6, max_length=6, description="Bill amounts, most recent first")
    PAY_AMT: list[float] = Field(min_length=6, max_length=6, description="Amounts paid, most recent first")

    @field_validator("PAY")
    @classmethod
    def _pay_range(cls, v):
        if any(x < -2 or x > 9 for x in v):
            raise ValueError("PAY values must be in [-2, 9]")
        return v

    @field_validator("PAY_AMT")
    @classmethod
    def _non_negative(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("PAY_AMT values must be >= 0")
        return v

    def to_frame(self) -> pd.DataFrame:
        row = {"LIMIT_BAL": self.LIMIT_BAL,
               "EDUCATION": 4 if self.EDUCATION in (0, 5, 6) else self.EDUCATION,
               "MARRIAGE": 3 if self.MARRIAGE == 0 else self.MARRIAGE,
               "AGE": self.AGE}
        row.update({f"PAY_{i+1}": v for i, v in enumerate(self.PAY)})
        row.update({f"BILL_AMT{i+1}": v for i, v in enumerate(self.BILL_AMT)})
        row.update({f"PAY_AMT{i+1}": v for i, v in enumerate(self.PAY_AMT)})
        return pd.DataFrame([row])[RAW_FEATURES]


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["review", "approve"]
    threshold: float
    model: str
    latency_ms: float


@lru_cache(maxsize=1)
def bundle():
    return joblib.load(C.MODELS_DIR / "model.joblib")


@app.get("/health")
def health():
    b = bundle()
    return {"status": "ok", "model": b["name"], "threshold": b["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(a: Applicant):
    t0 = time.perf_counter()
    b = bundle()
    raw = a.to_frame()
    X = engineer(raw) if b["engineered"] else raw
    p = float(b["model"].predict_proba(X[b["features"]])[:, 1][0])
    return Prediction(
        probability_of_default=round(p, 4),
        decision="review" if p >= b["threshold"] else "approve",
        threshold=b["threshold"],
        model=b["name"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
