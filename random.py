import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt

import dagshub
dagshub.init(repo_owner='fahimai001', repo_name='dagshub_project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/fahimai001/dagshub_project.mlflow")

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth = 10
n_estimators = 15
# Initialize the Decision Tree classifier
rf_classifier = RandomForestClassifier(random_state=42, max_depth=max_depth, n_estimators=n_estimators)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Log accuracy metric with MLflow
mlflow.set_experiment('fahim_run')
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)

    # Get the classification report and log individual metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    
    # Log precision, recall, and f1-score for class 0, 1, 2
    for class_label in range(3):  # There are 3 classes in Iris dataset (0, 1, 2)
        mlflow.log_metric(f"precision_class_{class_label}", report[str(class_label)]['precision'])
        mlflow.log_metric(f"recall_class_{class_label}", report[str(class_label)]['recall'])
        mlflow.log_metric(f"f1_class_{class_label}", report[str(class_label)]['f1-score'])
        
            
            # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names, ax=ax, cmap='viridis')
        plt.title("Confusion Matrix")
        
        # Save the plot to a file
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close(fig)

        # Log the plot as an artifact in MLflow
        mlflow.log_artifact(plot_path)
        
        mlflow.log_artifact(__file__)
        
        mlflow.sklearn.log_model(rf_classifier, 'random_forest')

        # Remove the local file after logging
        os.remove(plot_path)
