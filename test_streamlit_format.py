#!/usr/bin/env python3
"""
Test script to verify Streamlit format compatibility
"""

import requests

def test_streamlit_format():
    """Test the format that Streamlit UI sends"""
    print("🧪 Testing Streamlit UI Format Compatibility...")
    
    # This mimics what Streamlit UI sends
    streamlit_data = {
        "no_of_dependents": 2,
        "education": "Graduate",  # No leading space
        "self_employed": "No",    # No leading space
        "income_annum": 8000000,
        "loan_amount": 25000000,
        "loan_term": 15,
        "cibil_score": 750,
        "residential_assets_value": 5000000,
        "commercial_assets_value": 3000000,
        "luxury_assets_value": 2000000,
        "bank_asset_value": 1000000
    }
    
    # Test all endpoints
    endpoints = [
        ("predict", "🔮 Prediction"),
        ("explain", "📊 Explanation"),
        ("recommend", "💡 Recommendations")
    ]
    
    for endpoint, description in endpoints:
        print(f"\n{description}...")
        response = requests.post(f"http://localhost:8000/api/v1/{endpoint}", json=streamlit_data)
        
        if response.status_code == 200:
            result = response.json()
            if endpoint == "predict":
                print(f"   ✅ {result['prediction']} ({result['probability']:.1%})")
            elif endpoint == "explain":
                print(f"   ✅ Explanation generated for {result['prediction']}")
            elif endpoint == "recommend":
                print(f"   ✅ {len(result['recommendations'])} recommendations generated")
        else:
            print(f"   ❌ Error: {response.text}")
    
    print("\n🎉 All Streamlit format tests completed!")

if __name__ == "__main__":
    test_streamlit_format()
