import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from core.model_loader import model_loader
import logging

logger = logging.getLogger(__name__)

class LoanPredictor:
    def __init__(self):
        self.model_loader = model_loader
        self.object_columns = ['education', 'self_employed']
        
    def preprocess_data(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure all required columns are present
        required_columns = self.model_loader.get_feature_names()
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Reorder columns to match training data
        df = df[required_columns]
        
        # Apply object encoding to categorical columns
        object_encoder = self.model_loader.get_encoder('object_encoder')
        df[self.object_columns] = object_encoder.transform(df[self.object_columns])
        
        return df
    
    def predict_single(self, data: Dict[str, Any], model_name: str = 'random_forest') -> Dict[str, Any]:
        """Make prediction for a single data point"""
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Get model
            model = self.model_loader.get_model(model_name)
            
            # Make prediction
            prediction_proba = model.predict_proba(processed_data)[0]
            prediction = model.predict(processed_data)[0]
            
            # Convert prediction back to original labels
            target_encoder = self.model_loader.get_encoder('target_encoder')
            prediction_label = target_encoder.inverse_transform([prediction])[0]
            
            # Get probability for the predicted class
            prob = prediction_proba[prediction]
            
            # Determine confidence level
            confidence = self._get_confidence_level(prob)
            
            return {
                'prediction': prediction_label,
                'probability': float(prob),
                'confidence': confidence,
                'prediction_proba': prediction_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data: List[Dict[str, Any]], model_name: str = 'random_forest') -> List[Dict[str, Any]]:
        """Make predictions for batch data"""
        results = []
        for item in data:
            result = self.predict_single(item, model_name)
            results.append(result)
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Low"

# Global predictor instance
predictor = LoanPredictor()
