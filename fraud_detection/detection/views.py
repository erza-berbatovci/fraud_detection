from django.shortcuts import render
from .forms import TransactionForm
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from django.http import HttpResponse
from openpyxl import Workbook

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO
import base64


def fraud_detection_view(request):
    result = None
    graph_url = None
    model_path = os.path.join('fraud_model.pkl')

    # If there is a file uploaded via POST
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Load CSV file into DataFrame
            df = pd.read_csv(file)
            print("CSV data loaded:")
            print(df.head())

            # Select numeric columns for anomaly detection
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty:
                raise ValueError("The CSV file must contain at least one numeric column for anomaly detection.")

            print("Using the following columns for anomaly detection:", numeric_df.columns.tolist())

            # Check if model exists and is compatible with the new columns
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Check if column names match
                if set(model.feature_names_in_) != set(numeric_df.columns):
                    print("Column mismatch detected. Retraining the model with new columns.")
                    model = None  # Reset model to force retraining
            else:
                model = None

            # If model is None, train a new one
            if model is None:
                contamination_rate = min(0.05, 1.0 / len(numeric_df))
                model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
                model.fit(numeric_df)
                joblib.dump(model, model_path)
                print(f"Model trained with contamination rate: {contamination_rate} and saved.")
            else:
                print("Using loaded model...")

            # Perform anomaly detection
            df['anomaly'] = model.predict(numeric_df)
            print("Data with anomaly labels:")
            print(df['anomaly'].value_counts())  # Count of anomaly labels to verify detection

            # Identify anomalous columns for each row flagged as an anomaly
            anomalies = []
            for _, row in df[df['anomaly'] == -1].iterrows():
                row_anomalies = {'transaction_id': row.get('transaction_id', 'N/A'), 'anomalous_columns': []}

                # Check column-wise contributions to anomaly
                for column in numeric_df.columns:
                    without_column = numeric_df.drop(columns=[column])
                    temp_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                    temp_model.fit(without_column)
                    score_with_column = model.decision_function([row[numeric_df.columns]])
                    score_without_column = temp_model.decision_function([row[without_column.columns]])
                    
                    # If excluding the column significantly decreases the anomaly score, mark it as anomalous
                    if score_without_column > score_with_column:
                        row_anomalies['anomalous_columns'].append(column)

                anomalies.append(row_anomalies)

            result = anomalies
            print("Anomalies detected:", result)

            # Generate histogram for the first numeric column in the dataset and highlight anomalies
            first_numeric_col = numeric_df.columns[0]
            fig, ax = plt.subplots()
            normal_data = df[df['anomaly'] != -1][first_numeric_col]
            anomalous_data = df[df['anomaly'] == -1][first_numeric_col]

            # Plot normal data in blue and anomalies in red
            ax.hist(normal_data, bins=10, color='skyblue', edgecolor='black', label="Normal Data", alpha=0.7)
            ax.hist(anomalous_data, bins=10, color='red', edgecolor='black', label="Anomalies", alpha=0.7)
            ax.set_title(f'Distribution of {first_numeric_col}')
            ax.set_xlabel(first_numeric_col)
            ax.set_ylabel('Transaction Count')
            ax.legend()

            # Save and encode the plot image for HTML display
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

        except Exception as e:
            result = f'Error processing file: {e}'
            print(result)

    # Render template with results and graph
    return render(request, 'fraud_detection.html', {
        'form': TransactionForm(),
        'result': result,
        'graph_url': graph_url,
    })

# Additional export and plotting functions go here if needed

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
    # Fetch all anomalies from the database
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
