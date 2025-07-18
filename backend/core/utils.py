import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def validate_loan_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean loan input data"""
    required_fields = [
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate data types and ranges
    cleaned_data = data.copy()
    
    # Integer fields
    int_fields = ['no_of_dependents', 'loan_term', 'cibil_score']
    for field in int_fields:
        try:
            cleaned_data[field] = int(cleaned_data[field])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {field}: must be an integer")
    
    # Float fields
    float_fields = [
        'income_annum', 'loan_amount', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    for field in float_fields:
        try:
            cleaned_data[field] = float(cleaned_data[field])
            if cleaned_data[field] < 0:
                raise ValueError(f"Invalid value for {field}: must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {field}: must be a number")
    
    # String fields - normalize by adding leading space if missing
    if 'education' in cleaned_data:
        edu_val = cleaned_data['education']
        if edu_val in ['Graduate', 'Not Graduate']:
            cleaned_data['education'] = f" {edu_val}"
        elif edu_val not in [' Graduate', ' Not Graduate']:
            raise ValueError("Education must be 'Graduate' or 'Not Graduate'")
    
    if 'self_employed' in cleaned_data:
        emp_val = cleaned_data['self_employed']
        if emp_val in ['Yes', 'No']:
            cleaned_data['self_employed'] = f" {emp_val}"
        elif emp_val not in [' Yes', ' No']:
            raise ValueError("Self_employed must be 'Yes' or 'No'")
    
    # Range validations
    if not (0 <= cleaned_data['no_of_dependents'] <= 10):
        raise ValueError("Number of dependents must be between 0 and 10")
    
    if not (300 <= cleaned_data['cibil_score'] <= 900):
        raise ValueError("CIBIL score must be between 300 and 900")
    
    if not (1 <= cleaned_data['loan_term'] <= 30):
        raise ValueError("Loan term must be between 1 and 30 years")
    
    return cleaned_data

def format_currency(amount: float) -> str:
    """Format amount as Indian currency"""
    return f"₹{amount:,.0f}"

def get_sample_data() -> Dict[str, Any]:
    """Get sample data for template"""
    return {
        'no_of_dependents': 2,
        'education': ' Graduate',
        'self_employed': ' No',
        'income_annum': 8000000,
        'loan_amount': 25000000,
        'loan_term': 15,
        'cibil_score': 750,
        'residential_assets_value': 5000000,
        'commercial_assets_value': 3000000,
        'luxury_assets_value': 2000000,
        'bank_asset_value': 1000000
    }

def create_template_csv() -> str:
    """Create template CSV content"""
    sample_data = get_sample_data()
    df = pd.DataFrame([sample_data])
    return df.to_csv(index=False)

def log_request(endpoint: str, data: Dict[str, Any]):
    """Log API request"""
    logger.info(f"API Request to {endpoint}: {data}")

def log_response(endpoint: str, response: Dict[str, Any]):
    """Log API response"""
    logger.info(f"API Response from {endpoint}: {response}")

def handle_api_error(error: Exception, endpoint: str) -> Dict[str, Any]:
    """Handle API errors consistently"""
    logger.error(f"Error in {endpoint}: {str(error)}")
    return {
        'error': True,
        'message': str(error),
        'endpoint': endpoint
    }
