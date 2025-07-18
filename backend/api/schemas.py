from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd

class LoanInput(BaseModel):
    no_of_dependents: int
    education: str  # Accepts both "Graduate"/"Not Graduate" and " Graduate"/" Not Graduate"
    self_employed: str  # Accepts both "Yes"/"No" and " Yes"/" No"
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str

class ExplanationResponse(BaseModel):
    prediction: str
    probability: float
    shap_values: Dict[str, float]
    top_contributing_features: List[Dict[str, Any]]
    feature_impact: Dict[str, str]

class RecommendationResponse(BaseModel):
    current_prediction: str
    recommendations: List[Dict[str, Any]]
    potential_improvements: Dict[str, float]

class BatchPredictionRequest(BaseModel):
    data: List[LoanInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class TemplateResponse(BaseModel):
    columns: List[str]
    sample_data: Dict[str, Any]
