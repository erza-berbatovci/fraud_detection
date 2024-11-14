from django.shortcuts import render
from .forms import TransactionForm
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
import os
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

    return analysis

def fraud_detection_view(request):
    result = None
    graph_url = None
    dataset_analysis = None
    classification_report_str = None
    accuracy = None
    anomalies_count = 0  # Variable to store the count of anomalies detected

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Load CSV file into DataFrame
            df = pd.read_csv(file)

            # Analyze the dataset and prepare numeric features for Isolation Forest
            dataset_analysis = analyze_dataset(df)
            numeric_df = df.select_dtypes(include='number').drop(columns=['TransactionID', 'customer_zip_code_prefix'], errors='ignore')

            if numeric_df.empty:
                raise ValueError("The dataset must contain meaningful numeric columns for anomaly detection.")

            # Train or load Isolation Forest model
            if os.path.exists(isolation_model_path):
                isolation_model = joblib.load(isolation_model_path)
                if set(isolation_model.feature_names_in_) != set(numeric_df.columns):
                    isolation_model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
                    isolation_model.fit(numeric_df)
                    joblib.dump(isolation_model, isolation_model_path)
            else:
                isolation_model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
                isolation_model.fit(numeric_df)
                joblib.dump(isolation_model, isolation_model_path)

            # Perform anomaly detection with Isolation Forest
            df['anomaly'] = isolation_model.predict(numeric_df)
            anomalies = df[df['anomaly'] == -1]  # Filter only the anomalies
            anomalies_count = len(anomalies)  # Count the number of anomalies detected

            # Save anomalies for export or later use
            result = anomalies.to_dict(orient='records')
            request.session['anomalies_result'] = result

            # Optional: Generate a graph (code omitted for brevity)

        except Exception as e:
            result = f"An error occurred while processing the file: {e}"

    return render(request, 'fraud_detection.html', {
        'form': TransactionForm(),
        'result': result,
        'graph_url': graph_url,
        'dataset_analysis': dataset_analysis,
        'classification_report': classification_report_str,
        'accuracy': accuracy,
        'anomalies_count': anomalies_count,  # Pass anomalies count to template
    })

def export_to_csv(request):
    anomalies = FraudTransaction.objects.all()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="anomalies.csv"'
    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Amount', 'Time', 'Location'])
    for anomaly in anomalies:
        writer.writerow([anomaly.transaction_id, anomaly.amount, anomaly.time, anomaly.location])
    return response

def export_to_excel(request):
    anomalies = FraudTransaction.objects.all()
    wb = Workbook()
    ws = wb.active
    ws.title = "Anomalies"
    ws.append(['Transaction ID', 'Amount', 'Time', 'Location'])
    for anomaly in anomalies:
        ws.append([anomaly.transaction_id, anomaly.amount, anomaly.time, anomaly.location])
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="anomalies.xlsx"'
    wb.save(response)
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
