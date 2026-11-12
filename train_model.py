import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
class StudentPerformance:
    def __init__(self, path):
        try:
            # Reading dataset using the given path
            self.df = pd.read_csv(path)
            print("Dataset loaded successfully")
        except FileNotFoundError:
            print("Dataset file not found. Check the path.")
            raise
        except Exception as e:
            print("Error:", e)
            raise
    # Step 1: Dataset Information
    def information(self):
        print("\nFirst 5 Records:")
        print(self.df.head())
        print("\nDataset Shape:")
        print(self.df.shape)
        print("\nDataset Information:")
        self.df.info()
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nStatistical Summary:")
        print(self.df.describe())
    # Step 2: Data Cleaning and Preprocessing
    def clean_data(self):
        # Check duplicate records
        print("\nDuplicate Records:")
        print(self.df.duplicated().sum())
        # Remove duplicate records
        self.df = self.df.drop_duplicates().copy()
        # Check missing values
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        # Remove missing values
        self.df = self.df.dropna().copy()
        # Convert categorical data into numerical
        self.df['Extracurricular Activities'] = (
            self.df['Extracurricular Activities'].map({
                'Yes': 1,
                'No': 0
            })
        )
        # Remove any rows that could not be mapped
        self.df = self.df.dropna().copy()
        print("\nAfter Preprocessing:")
        print(self.df.head())
        print("\nUpdated Data Types:")
        print(self.df.dtypes)
        print("\nUpdated Dataset Shape:")
        print(self.df.shape)
        print("\nData Cleaning Completed")
    # Step 3: Exploratory Data Analysis
    def eda(self):
        # Performance Index Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(
            self.df['Performance Index'],
            kde=True
        )
        plt.title("Performance Index Distribution")
        plt.xlabel("Performance Index")
        plt.ylabel("Frequency")
        plt.show()
        # Correlation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            self.df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f'
        )
        plt.title("Correlation Heatmap")
        plt.show()
        # Hours Studied vs Performance Index
        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            x='Hours Studied',
            y='Performance Index',
            data=self.df
        )
        plt.title("Hours Studied vs Performance Index")
        plt.show()
        # Previous Scores vs Performance Index
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x='Previous Scores',
            y='Performance Index',
            data=self.df
        )
        plt.title("Previous Scores vs Performance Index")
        plt.show()
    # Step 4: Feature Selection
    def feature_selection(self):
        # Independent variables
        self.X = self.df.drop(
            'Performance Index',
            axis=1
        )
        # Dependent variable
        self.y = self.df['Performance Index']
        print("\nIndependent Variables (X):")
        print(self.X.head())
        print("\nDependent Variable (Y):")
        print(self.y.head())
        print("\nX Shape:", self.X.shape)
        print("Y Shape:", self.y.shape)
    # Step 5: Train-Test Split
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42
            )
        )
        print("\nTraining Records:", len(self.X_train))
        print("Testing Records:", len(self.X_test))
    # Step 6: Model Training
    def train_model(self):
        self.model = LinearRegression()
        # Train the model
        self.model.fit(
            self.X_train,
            self.y_train
        )
        print("\nModel trained successfully")
        print("\nIntercept:", self.model.intercept_)
        print("\nCoefficients:")
        print(self.model.coef_)
    # Step 7: Model Testing and Evaluation
    def evaluate_model(self):
        # Training predictions
        self.y_train_pred = self.model.predict(
            self.X_train
        )
        # Testing predictions
        self.y_test_pred = self.model.predict(
            self.X_test
        )
        # Training R2 Score
        train_accuracy = r2_score(
            self.y_train,
            self.y_train_pred
        )
        # Testing R2 Score
        test_accuracy = r2_score(
            self.y_test,
            self.y_test_pred
        )
        # Other evaluation metrics
        mae = mean_absolute_error(
            self.y_test,
            self.y_test_pred
        )
        mse = mean_squared_error(
            self.y_test,
            self.y_test_pred
        )
        rmse = np.sqrt(mse)
        print("\nModel Evaluation Results")
        print("Training R2 Score:", train_accuracy)
        print("Testing R2 Score:", test_accuracy)
        print("Training Score (%):", train_accuracy * 100)
        print("Testing Score (%):", test_accuracy * 100)
        print("\nMAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        # Actual vs Predicted
        result = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': self.y_test_pred
        })
        print("\nActual vs Predicted:")
        print(result.head(10))
        # Actual vs Predicted Graph
        plt.figure(figsize=(8, 5))
        plt.scatter(
            self.y_test,
            self.y_test_pred
        )
        plt.xlabel("Actual Performance Index")
        plt.ylabel("Predicted Performance Index")
        plt.title("Actual vs Predicted Performance")
        plt.show()
    # Step 8: Save Model using Pickle
    def save_model(self):
        try:
            with open("model.pkl", "wb") as file:
                pickle.dump(
                    self.model,
                    file
                )
            print("\nModel saved successfully as model.pkl")
        except Exception as e:
            print("Model Saving Error:", e)
            raise
# Main Program
if __name__ == "__main__":
    try:
        # Get project directory
        base_dir = Path(__file__).resolve().parent
        # Dataset path
        dataset_path = (
            base_dir / "DataSet" / "Student_Performance.csv"
        )
        # Create object
        obj = StudentPerformance(dataset_path)
        # Step 1
        obj.information()
        # Step 2
        obj.clean_data()
        # Step 3
        obj.eda()
        # Step 4
        obj.feature_selection()
        # Step 5
        obj.split_data()
        # Step 6
        obj.train_model()
        # Step 7
        obj.evaluate_model()
        # Step 8
        obj.save_model()
    except Exception as e:
        print("\nProject Execution Error:", e)
    finally:
        print("\nProject Execution Completed")
