from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import Response
import pandas as pd
import io
from typing import List, Dict, Any
import logging

from api.schemas import (
    LoanInput, PredictionResponse, ExplanationResponse, 
    RecommendationResponse, BatchPredictionRequest, BatchPredictionResponse,
    TemplateResponse
)
from core.predictor import predictor
from core.explainer import explainer
from core.recommender import recommender
from core.utils import validate_loan_input, create_template_csv, get_sample_data, log_request, log_response, handle_api_error

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_loan(loan_data: LoanInput):
    """Predict loan approval for single application"""
    try:
        # Convert to dict and validate
        data = loan_data.dict()
        validated_data = validate_loan_input(data)
        
        log_request("predict", validated_data)
        
        # Get prediction
        result = predictor.predict_single(validated_data)
        
        response = PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence']
        )
        
        log_response("predict", response.dict())
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predict loan approval for batch applications"""
    try:
        # Validate all inputs
        validated_data = []
        for item in batch_request.data:
            data = item.dict()
            validated_data.append(validate_loan_input(data))
        
        log_request("predict/batch", f"Batch size: {len(validated_data)}")
        
        # Get predictions
        results = predictor.predict_batch(validated_data)
        
        # Format response
        predictions = [
            PredictionResponse(
                prediction=result['prediction'],
                probability=result['probability'],
                confidence=result['confidence']
            )
            for result in results
        ]
        
        response = BatchPredictionResponse(predictions=predictions)
        log_response("predict/batch", f"Batch predictions completed: {len(predictions)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict loan approval from CSV file"""
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Validate and predict
        results = []
        for _, row in df.iterrows():
            try:
                data = row.to_dict()
                validated_data = validate_loan_input(data)
                result = predictor.predict_single(validated_data)
                results.append(result)
            except Exception as e:
                results.append({
                    'prediction': 'Error',
                    'probability': 0.0,
                    'confidence': 'Error',
                    'error': str(e)
                })
        
        # Create response DataFrame
        df_results = pd.DataFrame(results)
        
        # Return as CSV
        output = io.StringIO()
        df_results.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
        
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(loan_data: LoanInput):
    """Explain loan prediction using SHAP values"""
    try:
        # Convert to dict and validate
        data = loan_data.dict()
        validated_data = validate_loan_input(data)
        
        log_request("explain", validated_data)
        
        # Get explanation
        result = explainer.explain_prediction(validated_data)
        
        response = ExplanationResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            shap_values=result['shap_values'],
            top_contributing_features=result['top_contributing_features'],
            feature_impact=result['feature_impact']
        )
        
        log_response("explain", "Explanation generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(loan_data: LoanInput):
    """Get recommendations for loan approval improvement"""
    try:
        # Convert to dict and validate
        data = loan_data.dict()
        validated_data = validate_loan_input(data)
        
        log_request("recommend", validated_data)
        
        # Get recommendations
        result = recommender.generate_recommendations(validated_data)
        
        response = RecommendationResponse(
            current_prediction=result['current_prediction'],
            recommendations=result['recommendations'],
            potential_improvements=result['potential_improvements']
        )
        
        log_response("recommend", "Recommendations generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/template")
async def get_template():
    """Get CSV template for loan applications"""
    try:
        # Create template CSV
        csv_content = create_template_csv()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=loan_application_template.csv"}
        )
        
    except Exception as e:
        logger.error(f"Template generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/template/json", response_model=TemplateResponse)
async def get_template_json():
    """Get JSON template for loan applications"""
    try:
        sample_data = get_sample_data()
        feature_names = list(sample_data.keys())
        
        return TemplateResponse(
            columns=feature_names,
            sample_data=sample_data
        )
        
    except Exception as e:
        logger.error(f"Template JSON generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Loan prediction API is running"}

@router.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    try:
        return {
            "available_models": ["random_forest", "decision_tree"],
            "default_model": "random_forest",
            "feature_count": len(predictor.model_loader.get_feature_names()),
            "features": predictor.model_loader.get_feature_names()
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
