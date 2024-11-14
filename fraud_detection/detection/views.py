from django.shortcuts import render
from .forms import TransactionForm
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
import os
import csv
from django.http import HttpResponse
from openpyxl import Workbook
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

matplotlib.use('Agg')  # Set backend to Agg for compatibility with Django

# Define paths for the models
isolation_model_path = os.path.join('isolation_forest_model.pkl')
random_forest_model_path = os.path.join('random_forest_model.pkl')

def analyze_dataset(df):
    analysis = {
        'data_shape': df.shape,
        'null_values': df.isnull().values.any()
    }

    # Look for common fraud-related columns
    if 'Class' in df.columns:
        target_column = 'Class'
    elif 'isFraud' in df.columns:
        target_column = 'isFraud'
    elif 'FraudIndicator' in df.columns:
        target_column = 'FraudIndicator'
    else:
        target_column = None

    if target_column:
        unique_values = df[target_column].unique()
        analysis['unique_target_values'] = unique_values
        total_transactions = len(df)
        fraud_transactions = df[target_column].value_counts().get(1, 0)
        normal_transactions = df[target_column].value_counts().get(0, 0)

        analysis['percentage_non_fraudulent'] = f"{(normal_transactions / total_transactions) * 100:.3f}%" if total_transactions else "N/A"
        analysis['percentage_fraudulent'] = f"{(fraud_transactions / total_transactions) * 100:.3f}%" if total_transactions else "N/A"
        analysis['total_fraud_transactions'] = fraud_transactions
        analysis['total_normal_transactions'] = normal_transactions
    else:
        analysis['unique_target_values'] = "N/A"
        analysis['percentage_non_fraudulent'] = "N/A"
        analysis['percentage_fraudulent'] = "N/A"
        analysis['total_fraud_transactions'] = "N/A"
        analysis['total_normal_transactions'] = "N/A"

    return analysis, target_column

def fraud_detection_view(request):
    result = None
    graph_url = None
    scatter_url = None
    dataset_analysis = None
    classification_report_str = None
    accuracy = None
    anomalies_count = 0

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Load CSV file into DataFrame
            df = pd.read_csv(file)

            # Analyze dataset and detect target column
            dataset_analysis, target_column = analyze_dataset(df)
            numeric_df = df.select_dtypes(include='number').drop(columns=['TransactionID', 'customer_zip_code_prefix'], errors='ignore')

            if numeric_df.empty:
                raise ValueError("The dataset must contain meaningful numeric columns for anomaly detection.")

            # Train or load Isolation Forest model
            if os.path.exists(isolation_model_path):
                isolation_model = joblib.load(isolation_model_path)
                if set(isolation_model.feature_names_in_) != set(numeric_df.columns):
                    isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
                    isolation_model.fit(numeric_df)
                    joblib.dump(isolation_model, isolation_model_path)
            else:
                isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
                isolation_model.fit(numeric_df)
                joblib.dump(isolation_model, isolation_model_path)

            # Perform anomaly detection with Isolation Forest
            df['anomaly'] = isolation_model.predict(numeric_df)
            anomalies = df[df['anomaly'] == -1]
            anomalies_count = len(anomalies)
            result = anomalies.to_dict(orient='records')
            request.session['anomalies_result'] = result

            # Generate scatter plot
            feature_x = 'Time'
            feature_y = 'Amount'
            fig, ax = plt.subplots()
            normal_data = df[df['anomaly'] != -1]
            anomalous_data = df[df['anomaly'] == -1]
            ax.scatter(normal_data[feature_x], normal_data[feature_y], color='blue', label='Normal', alpha=0.6)
            ax.scatter(anomalous_data[feature_x], anomalous_data[feature_y], color='red', label='Anomaly', alpha=0.6)
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            ax.set_title(f'Scatter Plot of {feature_y} vs {feature_x} (Anomalies Highlighted)')
            ax.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            # Train and evaluate Random Forest if a target column is detected
            if target_column:
                X = numeric_df
                y = df[target_column]

                # Train-test split for supervised learning
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train Random Forest model
                random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
                random_forest_model.fit(X_train, y_train)
                
                # Make predictions and calculate metrics
                y_pred = random_forest_model.predict(X_test)
                classification_report_str = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)

        except Exception as e:
            result = f"An error occurred while processing the file: {e}"

    return render(request, 'fraud_detection.html', {
        'form': TransactionForm(),
        'result': result,
        'graph_url': graph_url,
        'scatter_url': scatter_url,
        'dataset_analysis': dataset_analysis,
        'classification_report': classification_report_str,
        'accuracy': accuracy,
        'anomalies_count': anomalies_count,
    })
def export_anomalies_to_csv(request):
    # Retrieve anomalies from session
    anomalies = request.session.get('anomalies_result', [])

    # If there are no anomalies, return an empty CSV file
    if not anomalies:
        return HttpResponse("No anomalies to export.", content_type="text/plain")

    # Create a CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="anomalies.csv"'

    # Write CSV headers
    writer = csv.writer(response)
    writer.writerow(anomalies[0].keys())  # Write headers based on keys of the first anomaly

    # Write anomaly data rows
    for anomaly in anomalies:
        writer.writerow(anomaly.values())

    return response


def plot_transaction_amounts(request):
    anomalies = FraudTransaction.objects.all()
    amounts = [anomaly.amount for anomaly in anomalies]

    all_data = pd.DataFrame(amounts, columns=['amount'])
    all_data['is_anomaly'] = all_data['amount'].apply(lambda x: x in amounts)

    normal_data = all_data[all_data['is_anomaly'] == False]['amount']
    anomalous_data = all_data[all_data['is_anomaly'] == True]['amount']

    fig, ax = plt.subplots()
    ax.hist(normal_data, bins=10, color='skyblue', edgecolor='black', label="Normal Data")
    ax.hist(anomalous_data, bins=10, color='red', edgecolor='black', label="Anomalies")

    ax.set_title('Transaction Amount Distribution')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Transaction Count')
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return render(request, 'plot_graph.html', {'graph_url': graph_url})
