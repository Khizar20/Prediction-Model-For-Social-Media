# Social Media Impact on Academic Performance Analysis
# This script analyzes the impact of social media usage on academic performance
# using machine learning techniques.

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import mlflow
import mlflow.sklearn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# Step 2: Load and prepare the dataset
def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for analysis
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Rename target column
    df = df.rename(columns={"Affects_Academic_Performance": "Affects"})
    
    # Drop rows with missing target
    df = df[df["Affects"].notna()]
    
    # Convert Yes/No to 1/0
    df["Affects"] = df["Affects"].map({"Yes": 1, "No": 0})
    
    return df

# Step 3: Feature Engineering
def engineer_features(df):
    """
    Create new features and transform existing ones
    """
    # Create interaction features
    df['Sleep_Usage_Interaction'] = df['Sleep_Hours_Per_Night'] * df['Avg_Daily_Usage_Hours']
    df['Mental_Usage_Interaction'] = df['Mental_Health_Score'] * df['Avg_Daily_Usage_Hours']
    
    # Bin continuous variables
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df[['Age_Binned', 'Usage_Binned', 'Sleep_Binned']] = discretizer.fit_transform(
        df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
    )
    
    return df

# Step 4: Prepare features and target
def prepare_features_target(df):
    """
    Prepare features and target variables
    """
    features = [
        "Age", "Gender", "Academic_Level", "Country",
        "Avg_Daily_Usage_Hours", "Most_Used_Platform",
        "Sleep_Hours_Per_Night", "Mental_Health_Score",
        "Sleep_Usage_Interaction", "Mental_Usage_Interaction",
        "Age_Binned", "Usage_Binned", "Sleep_Binned"
    ]
    
    X = df[features]
    y = df["Affects"]
    
    return X, y

# Step 5: Preprocess data
def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess the data including encoding, scaling, and train-test split
    """
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=["Gender", "Academic_Level", "Country", "Most_Used_Platform"], 
                             drop_first=False)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numeric features
    num_cols = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score",
                "Sleep_Usage_Interaction", "Mental_Usage_Interaction"]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler

# Step 6: Train and evaluate model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the Random Forest model
    """
    # Initialize model with balanced parameters
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, cv_scores, feature_importance, y_pred

# Step 7: Visualize results
def visualize_results(feature_importance, cv_scores):
    """
    Create visualizations of the results
    """
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Plot cross-validation scores
    plt.figure(figsize=(8, 4))
    plt.boxplot(cv_scores)
    plt.title('Cross-Validation Scores Distribution')
    plt.ylabel('Accuracy')
    plt.savefig('cv_scores.png')

# Step 8: Main execution
def main():
    # Set up MLflow
    mlflow.set_experiment("SocialMedia_AcademicImpact")
    
    with mlflow.start_run():
        # Load and prepare data
        df = load_and_prepare_data("Student_survey.csv")
        
        # Print class distribution
        print("Class distribution:")
        print(df["Affects"].value_counts(normalize=True))
        
        # Engineer features
        df = engineer_features(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # Train and evaluate model
        model, cv_scores, feature_importance, y_pred = train_and_evaluate_model(
            X_train, X_test, y_train, y_test
        )
        
        # Print results
        print("\nCross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())
        print("CV score std:", cv_scores.std())
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        print("\nTest set metrics:")
        print(classification_report(y_test, y_pred))
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        # Visualize results
        visualize_results(feature_importance, cv_scores)
        
        # Save model and scaler
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("model.pkl")
        
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("scaler.pkl")
        
        # Log visualizations
        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("cv_scores.png")

if __name__ == "__main__":
    main() 