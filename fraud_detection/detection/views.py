from django.shortcuts import render
from .forms import TransactionForm
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from django.http import HttpResponse
from openpyxl import Workbook
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for compatibility with Django
import matplotlib.pyplot as plt


# Define the model path
model_path = os.path.join('fraud_model.pkl')

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

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Load CSV file into DataFrame
            df = pd.read_csv(file)
            print("CSV data loaded successfully.")
            print(df.head())

            # Analyze the dataset
            dataset_analysis = analyze_dataset(df)
            
            # Automatically select numeric columns
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty:
                raise ValueError("The dataset must contain numeric columns for anomaly detection.")

            print("Using the following columns for anomaly detection:", numeric_df.columns.tolist())

            # Load or train the model
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Retrain model if columns do not match
                if set(model.feature_names_in_) != set(numeric_df.columns):
                    print("Column mismatch detected. Retraining the model with new columns.")
                    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
                    model.fit(numeric_df)
                    joblib.dump(model, model_path)
            else:
                print("Training a new model...")
                model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
                model.fit(numeric_df)
                joblib.dump(model, model_path)
                print("Model trained and saved.")

            # Perform anomaly detection
            df['anomaly'] = model.predict(numeric_df)
            anomalies = df[df['anomaly'] == -1]
            result = anomalies.to_dict(orient='records')
            print(f"Number of anomalies detected: {len(result)}")

            # Generate a histogram for the first numeric column
            first_numeric_col = numeric_df.columns[0]
            fig, ax = plt.subplots()
            normal_data = df[df['anomaly'] != -1][first_numeric_col]
            anomalous_data = anomalies[first_numeric_col]

            ax.hist(normal_data, bins=10, color='skyblue', edgecolor='black', label="Normal Data", alpha=0.7)
            ax.hist(anomalous_data, bins=10, color='red', edgecolor='black', label="Anomalies", alpha=0.7)
            ax.set_title(f'Distribution of {first_numeric_col}')
            ax.set_xlabel(first_numeric_col)
            ax.set_ylabel('Transaction Count')
            ax.legend()

            # Convert plot to base64 for HTML display
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

        except Exception as e:
            result = f"An error occurred while processing the file: {e}"
            print(result)

    return render(request, 'fraud_detection.html', {
        'form': TransactionForm(),
        'result': result,
        'graph_url': graph_url,
        'dataset_analysis': dataset_analysis,
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
