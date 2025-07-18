import pandas as pd
import numpy as np
from typing import Dict, Any, List
from core.explainer import explainer
import logging

logger = logging.getLogger(__name__)

class LoanRecommender:
    def __init__(self):
        self.explainer = explainer
        
    def generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for loan approval improvement"""
        try:
            # Get explanation first
            explanation = self.explainer.explain_prediction(data)
            
            # Only generate recommendations if loan is rejected
            if explanation['prediction'].strip() == 'Approved':
                return {
                    'current_prediction': explanation['prediction'],
                    'recommendations': [
                        {
                            'priority': 'High',
                            'feature': 'loan_approved',
                            'recommendation': 'Congratulations! Your loan has been approved. Continue maintaining your good financial profile.',
                            'actionable': True
                        }
                    ],
                    'potential_improvements': {}
                }
            
            # Generate recommendations based on SHAP values
            recommendations = self._generate_specific_recommendations(
                data, explanation['shap_values'], explanation['top_contributing_features']
            )
            
            # Calculate potential improvements
            potential_improvements = self._calculate_potential_improvements(
                data, explanation['shap_values']
            )
            
            return {
                'current_prediction': explanation['prediction'],
                'recommendations': recommendations,
                'potential_improvements': potential_improvements
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            raise
    
    def _generate_specific_recommendations(self, data: Dict[str, Any], shap_values: Dict[str, float], top_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific recommendations based on SHAP values"""
        recommendations = []
        
        # Sort features by negative impact (most negative first)
        negative_features = [(f['feature'], f['shap_value']) for f in top_features if f['shap_value'] < 0]
        negative_features.sort(key=lambda x: x[1])  # Sort by SHAP value (most negative first)
        
        for feature, shap_val in negative_features[:5]:  # Top 5 negative features
            current_value = data[feature]
            
            if feature == 'cibil_score':
                if current_value < 550:
                    recommendations.append({
                        'priority': 'High',
                        'feature': feature,
                        'current_value': current_value,
                        'recommendation': f"Improve CIBIL score from {current_value} to 550+ for better approval chances",
                        'impact': abs(shap_val),
                        'actionable': True
                    })
            
            elif feature == 'income_annum':
                target_income = current_value * 1.2  # 20% increase
                recommendations.append({
                    'priority': 'Medium',
                    'feature': feature,
                    'current_value': current_value,
                    'recommendation': f"Increase annual income from ₹{current_value:,.0f} to ₹{target_income:,.0f}",
                    'impact': abs(shap_val),
                    'actionable': True
                })
            
            elif feature == 'loan_amount':
                if current_value > data['income_annum'] * 3:  # High loan-to-income ratio
                    target_amount = data['income_annum'] * 3
                    recommendations.append({
                        'priority': 'High',
                        'feature': feature,
                        'current_value': current_value,
                        'recommendation': f"Reduce loan amount from ₹{current_value:,.0f} to ₹{target_amount:,.0f} (3x annual income)",
                        'impact': abs(shap_val),
                        'actionable': True
                    })
            
            elif feature == 'loan_term':
                if current_value > 15:
                    recommendations.append({
                        'priority': 'Medium',
                        'feature': feature,
                        'current_value': current_value,
                        'recommendation': f"Consider reducing loan term from {current_value} to 10-15 years",
                        'impact': abs(shap_val),
                        'actionable': True
                    })
            
            elif feature == 'bank_asset_value':
                target_assets = current_value * 1.5
                recommendations.append({
                    'priority': 'Medium',
                    'feature': feature,
                    'current_value': current_value,
                    'recommendation': f"Increase bank assets from ₹{current_value:,.0f} to ₹{target_assets:,.0f}",
                    'impact': abs(shap_val),
                    'actionable': True
                })
            
            elif feature == 'residential_assets_value':
                target_assets = current_value * 1.3
                recommendations.append({
                    'priority': 'Low',
                    'feature': feature,
                    'current_value': current_value,
                    'recommendation': f"Increase residential assets from ₹{current_value:,.0f} to ₹{target_assets:,.0f}",
                    'impact': abs(shap_val),
                    'actionable': False
                })
        
        # Return only data-driven recommendations based on actual model features
        return recommendations
    
    def _calculate_potential_improvements(self, data: Dict[str, Any], shap_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate potential probability improvements"""
        improvements = {}
        
        # Calculate potential improvement for key features
        for feature, shap_val in shap_values.items():
            if shap_val < -0.05:  # Significantly negative impact
                if feature == 'cibil_score':
                    # Estimate improvement if CIBIL score is increased to 750
                    current_score = data[feature]
                    if current_score < 750:
                        score_improvement = (750 - current_score) / 900  # Normalize improvement
                        improvements[feature] = abs(shap_val) * score_improvement
                
                elif feature == 'loan_amount':
                    # Estimate improvement if loan amount is reduced
                    current_amount = data[feature]
                    income = data['income_annum']
                    if current_amount > income * 3:
                        amount_reduction = (current_amount - income * 3) / current_amount
                        improvements[feature] = abs(shap_val) * amount_reduction
                
                elif feature == 'income_annum':
                    # Estimate improvement with 20% income increase
                    improvements[feature] = abs(shap_val) * 0.2
        
        return improvements

# Global recommender instance
recommender = LoanRecommender()
