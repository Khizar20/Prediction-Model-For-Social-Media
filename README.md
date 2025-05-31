# Social Media Academic Impact Predictor

This project uses machine learning to predict whether social media usage is likely to affect academic performance based on various factors such as usage patterns, sleep habits, and mental health indicators.

## Features

- Predicts impact of social media on academic performance
- Analyzes multiple factors including:
  - Social media usage patterns
  - Sleep habits
  - Mental health indicators
  - Academic level
  - Demographics
- Provides detailed analysis and recommendations
- Interactive web interface using Streamlit

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py                 # Streamlit web application
├── model.pkl             # Trained Random Forest model
├── scaler.pkl            # Feature scaler
├── Student_survey.csv    # Dataset
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Input your information:
   - Age
   - Gender
   - Academic Level
   - Country
   - Average Daily Social Media Usage
   - Most Used Platform
   - Sleep Hours
   - Mental Health Score

4. Click "Predict" to get the analysis

## Model Details

The prediction model uses a Random Forest classifier with the following features:
- Demographics (Age, Gender, Academic Level, Country)
- Usage Patterns (Daily Usage Hours, Most Used Platform)
- Health Indicators (Sleep Hours, Mental Health Score)
- Interaction Features (Sleep-Usage, Mental-Usage)
- Binned Features (Age, Usage, Sleep patterns)

## Output Interpretation

The model provides:
- Prediction (Likely/Not Likely to affect academic performance)
- Confidence score
- Identified risk factors
- Specific recommendations
- Positive factors in your profile
- Suggestions for maintaining balance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Student Survey on Social Media Usage
- Libraries: Streamlit, scikit-learn, pandas, numpy
- Contributors: [Your Name/Team]

## Contact

For questions or feedback, please open an issue in the repository. 