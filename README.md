# 🏦 Loan Prediction System

A comprehensive **machine learning-powered** loan approval prediction system built with **FastAPI** backend and **Streamlit** frontend. This system provides intelligent credit risk assessment with SHAP explanations and actionable recommendations for loan applications.

### 🌐 Online Deployment
The Loan Prediction System is live and accessible at:
**[https://render-loansense-web.onrender.com/](https://render-loansense-web.onrender.com/)**

### 📹 Video Demonstration
Watch the complete system walkthrough and deployment guide:
**[YouTube Demo Video](https://youtu.be/3p2w_T2Gn50)**


## 🌟 Overview

This system helps financial institutions automate loan approval decisions using machine learning models. It features:

- **🤖 Smart Predictions**: ML models (Random Forest & Decision Tree) for accurate loan approval predictions
- **📊 SHAP Explanations**: Transparent AI with feature importance analysis
- **💡 Recommendations**: Actionable suggestions to improve loan approval chances
- **🚀 FastAPI Backend**: High-performance REST API with comprehensive documentation
- **🎨 Streamlit Frontend**: User-friendly web interface for single and batch predictions
- **📁 Batch Processing**: Handle multiple applications via CSV upload or manual entry

## 🏗️ Architecture

```
📁 Loan Prediction System
├── 🖥️  Backend (FastAPI)           # Core ML API Server
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── endpoints.py          # API route definitions
│   │   └── schemas.py            # Pydantic data models
│   ├── core/
│   │   ├── model_loader.py       # ML model management
│   │   ├── predictor.py          # Prediction engine
│   │   ├── explainer.py          # SHAP explanations
│   │   ├── recommender.py        # Improvement recommendations
│   │   └── utils.py              # Utility functions
│   └── models/                   # Trained ML models & encoders
│       ├── enc1_random_forest.pkl
│       ├── enc1_decision_tree.pkl
│       └── encoder/
├── 🎨 Frontend (Streamlit)         # Web User Interface
│   └── streamlit_ui.py           # Interactive dashboard
├── 📊 Data & Models
│   ├── data/                     # Training and test datasets
│   ├── models/                   # Production ML models
│   └── notebooks/                # Jupyter notebooks for EDA & modeling
├── 🛠️  Scripts
│   ├── setup.sh                  # Environment setup
│   ├── run_backend.sh           # Backend server launcher
│   ├── run_frontend.sh          # Frontend app launcher
│   └── test_api.py              # API testing suite
└── 📚 Documentation
    ├── README.md                # This file
    ├── QUICKSTART.md           # Quick start guide
    └── DEPLOYMENT_SUCCESS.md   # Deployment notes
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda environment (recommended: `stable`)
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Loan_prediction
```

### 2. Environment Setup
```bash
# Make scripts executable
chmod +x setup.sh run_backend.sh run_frontend.sh

# Setup environment and start both services
./setup.sh
```

### 3. Manual Setup (Alternative)

#### Backend Setup
```bash
cd backend
conda activate stable  # or your preferred environment
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup (New Terminal)
```bash
# Install frontend dependencies
pip install streamlit plotly requests pandas
streamlit run streamlit_ui.py --server.port 8501
```

### 4. Access Applications
- **🔗 API Documentation**: http://localhost:8000/docs
- **🎨 Streamlit Dashboard**: http://localhost:8501  
- **💊 Health Check**: http://localhost:8000/api/v1/health

## 📡 API Endpoints

### Core Prediction APIs
| Endpoint | Method | Description | Input | Output |
|----------|--------|-------------|-------|--------|
| `/api/v1/predict` | POST | Single loan prediction | Loan application data | Prediction + probability |
| `/api/v1/predict/batch` | POST | Multiple predictions | Array of applications | Batch predictions |
| `/api/v1/predict/csv` | POST | CSV file upload | CSV file | Predictions for all rows |
| `/api/v1/explain` | POST | SHAP explanations | Loan application data | Feature importance analysis |
| `/api/v1/recommend` | POST | Improvement tips | Loan application data | Actionable recommendations |

### Utility APIs
| Endpoint | Method | Description | Output |
|----------|--------|-------------|--------|
| `/api/v1/health` | GET | System health check | API status |
| `/api/v1/models` | GET | Available models info | Model metadata |
| `/api/v1/template` | GET | Download CSV template | CSV template file |

### 🔧 Example API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 8000000,
    "loan_amount": 25000000,
    "loan_term": 15,
    "cibil_score": 750,
    "residential_assets_value": 5000000,
    "commercial_assets_value": 3000000,
    "luxury_assets_value": 2000000,
    "bank_asset_value": 1000000
  }'
```

#### Response
```json
{
  "prediction": "Approved",
  "probability": 0.85,
  "confidence": "High"
}
```

#### SHAP Explanation
```bash
curl -X POST "http://localhost:8000/api/v1/explain" \
  -H "Content-Type: application/json" \
  -d '{...same data as above...}'
```

## 🎯 Features

### ✅ Backend Features (FastAPI)
- **🤖 Multiple ML Models**: Random Forest (primary) & Decision Tree
- **🔍 Flexible Predictions**: Single, batch, and CSV file processing
- **📊 SHAP Integration**: Transparent AI with feature importance
- **💡 Smart Recommendations**: AI-driven improvement suggestions
- **📁 File Operations**: CSV template generation and bulk processing
- **🛡️ Input Validation**: Comprehensive data validation with Pydantic
- **📝 Logging System**: Detailed request/response logging
- **🔄 CORS Support**: Cross-origin requests enabled
- **📚 Auto Documentation**: Interactive API docs with Swagger UI

### ✅ Frontend Features (Streamlit)
- **📋 Single Prediction Form**: Interactive loan application form
- **📊 Batch Processing**: Multiple applications at once
- **📁 CSV Upload Interface**: Drag-and-drop file processing
- **📈 SHAP Visualizations**: Interactive waterfall plots
- **💡 Visual Recommendations**: Color-coded improvement suggestions
- **🔧 System Monitoring**: Real-time API health dashboard
- **📱 Responsive Design**: Multi-column layouts for better UX
- **🎨 Modern UI**: Custom CSS styling with professional appearance

## 📊 Input Features

| Feature | Type | Description | Example | Range/Options |
|---------|------|-------------|---------|---------------|
| `no_of_dependents` | Integer | Number of dependents | 2 | 0-10 |
| `education` | String | Education level | "Graduate" | "Graduate", "Not Graduate" |
| `self_employed` | String | Employment status | "No" | "Yes", "No" |
| `income_annum` | Float | Annual income (₹) | 8000000 | > 0 |
| `loan_amount` | Float | Requested loan (₹) | 25000000 | > 0 |
| `loan_term` | Integer | Loan duration (years) | 15 | 1-30 |
| `cibil_score` | Integer | Credit score | 750 | 300-900 |
| `residential_assets_value` | Float | Residential assets (₹) | 5000000 | ≥ 0 |
| `commercial_assets_value` | Float | Commercial assets (₹) | 3000000 | ≥ 0 |
| `luxury_assets_value` | Float | Luxury assets (₹) | 2000000 | ≥ 0 |
| `bank_asset_value` | Float | Bank deposits (₹) | 1000000 | ≥ 0 |

## 🔍 Model Performance & Methodology

### Machine Learning Models
- **🌳 Random Forest**: Primary model with ensemble learning for robust predictions
- **🌲 Decision Tree**: Alternative model for interpretable decision-making
- **📊 Feature Engineering**: Automated preprocessing and encoding
- **🎯 Cross-Validation**: Model validation for reliable performance metrics

### SHAP Integration
- **🔍 Feature Importance**: Individual feature impact on predictions
- **📈 Visualizations**: Waterfall plots showing decision reasoning
- **🎯 Transparency**: Explainable AI for regulatory compliance

## 💡 Recommendations System

The system provides tiered recommendations based on SHAP feature importance:

### 🔴 **High Priority Recommendations**
- **CIBIL Score Improvement**: Most critical factor
- **Loan Amount Optimization**: Reduce requested amount
- **Debt-to-Income Ratio**: Balance loan amount with income

### 🟡 **Medium Priority Recommendations**  
- **Income Enhancement**: Increase annual income
- **Loan Term Adjustment**: Optimize repayment period
- **Employment Stability**: Consider employment status

### 🟢 **Low Priority Recommendations**
- **Asset Value Improvements**: Increase collateral assets
- **Dependent Optimization**: Family size considerations

## 🛠️ Development & Testing

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development  
```bash
streamlit run streamlit_ui.py --server.port 8501
```

### API Testing
```bash
# Run comprehensive API tests
python test_api.py

# Test Streamlit format compatibility
python test_streamlit_format.py
```

### Adding New Models
1. Train and save model as `.pkl` file in `backend/models/`
2. Update `model_loader.py` to include new model
3. Modify prediction endpoints to support new model
4. Update model information in `/api/v1/models` endpoint

## 📋 Dependencies

### Backend Requirements
```
fastapi==0.104.1
uvicorn==0.24.0
pandas>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
shap>=0.43.0
joblib>=1.3.0
pydantic==2.5.0
python-multipart==0.0.6
```

### Frontend Requirements
```
streamlit
plotly
requests
pandas
```

## 🚀 Deployment

### Production Considerations
- **🔒 Security**: Update CORS settings for production
- **📊 Monitoring**: Implement comprehensive logging
- **⚡ Performance**: Consider model caching strategies
- **🔄 Scaling**: Use multiple workers for FastAPI
- **💾 Database**: Consider database integration for persistence

### Docker Deployment (Future Enhancement)
```dockerfile
# Example Dockerfile structure
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning models
- **SHAP** for explainable AI capabilities
- **FastAPI** for high-performance API framework
- **Streamlit** for rapid web app development
- **Plotly** for interactive visualizations

## 📞 Support

For support, please:
1. Check the API documentation at `/docs`
2. Review the troubleshooting section
3. Open an issue on GitHub
4. Contact the development team

---

**Made with ❤️ for transparent and intelligent loan prediction**

A complete **backend-first** credit risk prediction system using **FastAPI** and **Streamlit**. This system provides ML-powered loan approval predictions with SHAP explanations and actionable recommendations.

## 🏗️ Architecture

```
📁 Project Structure
├── backend/                    # FastAPI Backend
│   ├── main.py                # Entry point
│   ├── api/
│   │   ├── endpoints.py       # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── core/
│   │   ├── model_loader.py    # Model loading
│   │   ├── predictor.py       # Prediction logic
│   │   ├── explainer.py       # SHAP explanations
│   │   ├── recommender.py     # Recommendations
│   │   └── utils.py           # Utilities
│   ├── models/                # ML models & encoders
│   │   ├── enc1_random_forest.pkl
│   │   ├── enc1_decision_tree.pkl
│   │   └── encoder/
│   └── requirements.txt
├── streamlit_ui.py            # Streamlit frontend
├── requirements.txt           # Frontend dependencies
├── setup.sh                  # Setup script
├── run_backend.sh            # Backend runner
└── run_frontend.sh           # Frontend runner
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
./setup.sh
```

### 2. Start Backend (Terminal 1)
```bash
./run_backend.sh
```

### 3. Start Frontend (Terminal 2)
```bash
./run_frontend.sh
```

### 4. Access Applications
- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **API Health Check**: http://localhost:8000/api/v1/health

## 📡 API Endpoints

### Core Endpoints
- `POST /api/v1/predict` - Single loan prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `POST /api/v1/predict/csv` - CSV upload predictions
- `POST /api/v1/explain` - SHAP explanations
- `POST /api/v1/recommend` - Improvement recommendations
- `GET /api/v1/template` - Download CSV template
- `GET /api/v1/health` - Health check
- `GET /api/v1/models` - Model information

### Example API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 8000000,
    "loan_amount": 25000000,
    "loan_term": 15,
    "cibil_score": 750,
    "residential_assets_value": 5000000,
    "commercial_assets_value": 3000000,
    "luxury_assets_value": 2000000,
    "bank_asset_value": 1000000
  }'
```

#### SHAP Explanation
```bash
curl -X POST "http://localhost:8000/api/v1/explain" \
  -H "Content-Type: application/json" \
  -d '{...same data as above...}'
```

## 🎯 Features

### ✅ Backend (FastAPI)
- **🤖 ML Models**: Random Forest & Decision Tree
- **🔍 Predictions**: Single, batch, and CSV upload
- **📊 SHAP Explanations**: Feature importance analysis
- **💡 Recommendations**: Actionable improvement suggestions
- **📁 File Processing**: CSV template & batch processing
- **🛡️ Validation**: Input validation and error handling
- **📝 Logging**: Comprehensive logging system
- **🔄 CORS**: Cross-origin resource sharing enabled

### ✅ Frontend (Streamlit)
- **📋 Single Prediction**: Manual input form
- **📊 Batch Processing**: Multiple applications at once
- **📁 CSV Upload**: File upload for bulk predictions
- **📈 SHAP Visualization**: Interactive feature importance plots
- **💡 Recommendations**: Visual improvement suggestions
- **🔧 API Status**: Health monitoring dashboard
- **📱 Responsive Design**: Multi-column layouts

## 📊 Input Features

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `no_of_dependents` | Integer | Number of dependents | 2 |
| `education` | String | Education level | "Graduate" |
| `self_employed` | String | Employment status | "No" |
| `income_annum` | Float | Annual income (₹) | 8000000 |
| `loan_amount` | Float | Requested loan amount (₹) | 25000000 |
| `loan_term` | Integer | Loan term in years | 15 |
| `cibil_score` | Integer | Credit score (300-900) | 750 |
| `residential_assets_value` | Float | Residential assets (₹) | 5000000 |
| `commercial_assets_value` | Float | Commercial assets (₹) | 3000000 |
| `luxury_assets_value` | Float | Luxury assets (₹) | 2000000 |
| `bank_asset_value` | Float | Bank assets (₹) | 1000000 |

## 🔍 Model Performance

- **Random Forest**: Primary model with cross-validation
- **Decision Tree**: Alternative model for comparison
- **SHAP Integration**: Feature importance explanations
- **Preprocessing**: Automated encoding and validation

## 💡 Recommendations System

The system provides actionable recommendations based on SHAP values:

- **🔴 High Priority**: CIBIL score improvement, loan amount reduction
- **🟡 Medium Priority**: Income increase, loan term adjustment
- **🟢 Low Priority**: Asset value improvements

## 🛠️ Development

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
streamlit run streamlit_ui.py --server.port 8501
```

### Adding New Models
1. Save model as `.pkl` file in `backend/models/`
2. Update `model_loader.py` to load the new model
3. Add model option to prediction endpoints

## 📝 API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with:
- **Swagger UI**: Interactive API testing
- **Request/Response Examples**: Sample data formats
- **Authentication**: Future auth implementation guide

## 🔧 Configuration

### Environment Variables
```bash
# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Frontend Configuration
STREAMLIT_SERVER_PORT=8501
API_BASE_URL=http://localhost:8000/api/v1
```

### Model Paths
- Models: `backend/models/`
- Encoders: `backend/models/encoder/`
- Logs: `backend/loan_prediction_api.log`

## 🚀 Deployment

### Docker (Future Enhancement)
```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- **Security**: Add authentication and rate limiting
- **Monitoring**: Implement health checks and metrics
- **Scaling**: Use gunicorn for production deployment
- **Database**: Add database for logging and analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues
1. **Models not loading**: Check file paths in `model_loader.py`
2. **API connection error**: Ensure backend is running on port 8000
3. **SHAP errors**: Verify model compatibility with SHAP
4. **CSV upload issues**: Check column names match exactly

### Support
- Check logs in `backend/loan_prediction_api.log`
- Visit API health endpoint: `/api/v1/health`
- Review API documentation: `/docs`

---

**Built with ❤️ using FastAPI, Streamlit, and scikit-learn**
