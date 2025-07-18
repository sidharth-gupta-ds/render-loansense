import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from core.model_loader import model_loader
from core.predictor import predictor
import logging

logger = logging.getLogger(__name__)

class LoanExplainer:
    def __init__(self):
        self.model_loader = model_loader
        self.predictor = predictor
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer"""
        try:
            model = self.model_loader.get_model('random_forest')
            self.explainer = self.model_loader.get_model('shap')
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            raise
    
    def explain_prediction(self, data: Dict[str, Any], model_name: str = 'random_forest') -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction"""
        try:
            # Get prediction first
            prediction_result = self.predictor.predict_single(data, model_name)
            
            # Preprocess data
            processed_data = self.predictor.preprocess_data(data)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(processed_data)
            logger.info(f"SHAP values type: {type(shap_values)}")
            logger.info(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'No shape'}")
            
            # For loan approval, we always want to show SHAP values for the "Approved" class (class 0)
            # This shows how each feature contributes to the probability of approval
            # 0 = Approved, 1 = Rejected
            approval_class = 0  # Always use class 0 (Approved) for interpretation
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'shape'):
                if len(shap_values.shape) == 3:
                    # Shape is (n_samples, n_features, n_classes)
                    # Use the SHAP values for the approval class (class 0)
                    shap_values_for_prediction = shap_values[0, :, approval_class]
                elif len(shap_values.shape) == 2:
                    # Shape is (n_samples, n_features)
                    shap_values_for_prediction = shap_values[0, :]
                else:
                    # Shape is (n_features,)
                    shap_values_for_prediction = shap_values
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                # List format for binary classification
                shap_values_for_prediction = shap_values[approval_class]
                if hasattr(shap_values_for_prediction, 'shape') and len(shap_values_for_prediction.shape) > 1:
                    shap_values_for_prediction = shap_values_for_prediction[0]
            else:
                shap_values_for_prediction = shap_values
                if hasattr(shap_values_for_prediction, 'shape') and len(shap_values_for_prediction.shape) > 1:
                    shap_values_for_prediction = shap_values_for_prediction[0]
            
            # Get feature names
            feature_names = self.model_loader.get_feature_names()
            
            # Create SHAP values dictionary
            shap_dict = {
                feature_names[i]: float(shap_values_for_prediction[i]) 
                for i in range(len(feature_names))
            }
            
            # Get top contributing features
            top_features = self._get_top_features(shap_dict, processed_data.iloc[0])
            
            # Get feature impact description
            feature_impact = self._get_feature_impact(shap_dict)
            
            return {
                'prediction': prediction_result['prediction'],
                'probability': prediction_result['probability'],
                'shap_values': shap_dict,
                'top_contributing_features': top_features,
                'feature_impact': feature_impact
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            raise
    
    def _get_top_features(self, shap_values: Dict[str, float], data_row: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top contributing features"""
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_features = []
        for i, (feature, shap_val) in enumerate(sorted_features[:top_n]):
            top_features.append({
                'rank': i + 1,
                'feature': feature,
                'shap_value': shap_val,
                'feature_value': float(data_row[feature]),
                'impact': 'Positive' if shap_val > 0 else 'Negative',
                'importance': abs(shap_val)
            })
        
        return top_features
    
    def _get_feature_impact(self, shap_values: Dict[str, float]) -> Dict[str, str]:
        """Get feature impact descriptions"""
        impact_descriptions = {}
        
        for feature, shap_val in shap_values.items():
            if abs(shap_val) > 0.1:  # Only describe significant impacts
                if shap_val > 0:
                    impact_descriptions[feature] = f"Increases approval probability by {abs(shap_val):.3f}"
                else:
                    impact_descriptions[feature] = f"Decreases approval probability by {abs(shap_val):.3f}"
        
        return impact_descriptions

# Global explainer instance
explainer = LoanExplainer()
