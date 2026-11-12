import pickle
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request
class StudentPerformanceApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.base_dir = Path(__file__).resolve().parent
        self.model_path = self.base_dir / "model.pkl"
        self.model = None
        self.load_model()
        self.app.add_url_rule(
            "/", "home", self.home, methods=["GET", "POST"]
        )
    def load_model(self):
        """Load the trained machine learning model."""
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Error: model.pkl not found. Run train_model.py first.")
        except Exception as error:
            print("Error while loading model:", error)
    def home(self):
        prediction = None
        error = None
        if request.method == "POST":
            try:
                if self.model is None:
                    raise RuntimeError(
                        "Model is not available. Please run train_model.py first."
                    )
                # Read values from the HTML form
                hours_studied = float(request.form["hours_studied"])
                previous_scores = float(request.form["previous_scores"])
                extracurricular = int(request.form["extracurricular"])
                sleep_hours = float(request.form["sleep_hours"])
                papers = float(request.form["papers"])
                # Basic input validation
                if not 0 <= hours_studied <= 24:
                    raise ValueError("Hours studied must be between 0 and 24.")
                if not 0 <= previous_scores <= 100:
                    raise ValueError("Previous scores must be between 0 and 100.")
                if extracurricular not in [0, 1]:
                    raise ValueError("Please select a valid extracurricular option.")
                if not 0 <= sleep_hours <= 24:
                    raise ValueError("Sleep hours must be between 0 and 24.")
                if not 0 <= papers <= 100:
                    raise ValueError("Sample question papers must be between 0 and 100.")
                # Prepare input in the same feature order used during training
                input_data = pd.DataFrame({
                    "Hours Studied": [hours_studied],
                    "Previous Scores": [previous_scores],
                    "Extracurricular Activities": [extracurricular],
                    "Sleep Hours": [sleep_hours],
                    "Sample Question Papers Practiced": [papers]
                })
                # Predict student performance
                result = self.model.predict(input_data)
                prediction = round(float(result[0]), 2)
            except ValueError as value_error:
                error = str(value_error)
            except KeyError:
                error = "Please fill in all the required fields."
            except Exception as exception:
                error = f"Prediction failed: {exception}"
        return render_template(
            "index.html",
            prediction=prediction,
            error=error
        )
# Create Flask application
student_app = StudentPerformanceApp()
app = student_app.app
if __name__ == "__main__":
    app.run(debug=True)
