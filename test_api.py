#!/usr/bin/env python3
"""
Test script for Loan Prediction API
"""

import requests
import json
import sys

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Test data
sample_loan_data = {
    "no_of_dependents": 2,
    "education": " Graduate",
    "self_employed": " No",
    "income_annum": 8000000,
    "loan_amount": 25000000,
    "loan_term": 15,
    "cibil_score": 750,
    "residential_assets_value": 5000000,
    "commercial_assets_value": 3000000,
    "luxury_assets_value": 2000000,
    "bank_asset_value": 1000000
}

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=sample_loan_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction test passed: {result['prediction']} ({result['probability']:.2%})")
            return True
        else:
            print(f"❌ Prediction test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def test_explanation():
    """Test explanation endpoint"""
    try:
        response = requests.post(f"{API_BASE_URL}/explain", json=sample_loan_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Explanation test passed: {len(result['shap_values'])} features explained")
            return True
        else:
            print(f"❌ Explanation test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Explanation test error: {e}")
        return False

def test_recommendation():
    """Test recommendation endpoint"""
    try:
        response = requests.post(f"{API_BASE_URL}/recommend", json=sample_loan_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recommendation test passed: {len(result['recommendations'])} recommendations")
            return True
        else:
            print(f"❌ Recommendation test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Recommendation test error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model info test passed: {len(result['available_models'])} models available")
            return True
        else:
            print(f"❌ Model info test failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Model info test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Loan Prediction API...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Prediction", test_prediction),
        ("Explanation", test_explanation),
        ("Recommendation", test_recommendation),
        ("Model Info", test_model_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the API server.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
