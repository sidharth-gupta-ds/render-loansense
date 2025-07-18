import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.encoders = {}
        self.feature_names = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ]
        self.load_models()
    
    def load_models(self):
        """Load all models and encoders"""
        try:
            # Load models
            self.models['random_forest'] = joblib.load(
                self.model_dir / 'enc1_random_forest.pkl'
            )
            self.models['decision_tree'] = joblib.load(
                self.model_dir / 'enc1_decision_tree.pkl'
            )
            
            # Load encoders
            self.encoders['object_encoder'] = joblib.load(
                self.model_dir / 'encoder' / 'enc1_object_encoder.pkl'
            )
            self.encoders['target_encoder'] = joblib.load(
                self.model_dir / 'encoder' / 'enc1_target_encoder.pkl'
            )
            
            logger.info("Models and encoders loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model(self, model_name: str = 'random_forest'):
        """Get a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]
    
    def get_encoder(self, encoder_name: str):
        """Get a specific encoder"""
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder {encoder_name} not found")
        return self.encoders[encoder_name]
    
    def get_feature_names(self):
        """Get feature names"""
        return self.feature_names.copy()

# Global model loader instance
model_loader = ModelLoader()
