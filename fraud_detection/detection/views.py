from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
import pandas as pd
import joblib
import os
import csv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from .models import Dataset, UserActivityLog  # Import custom models
from .forms import TransactionForm
from django.contrib.auth.forms import UserCreationForm

# Set backend for matplotlib
plt.switch_backend('Agg')

# Paths for the models
ISOLATION_MODEL_PATH = os.path.join('isolation_forest_model.pkl')

# Helper function to check if the user is an admin
def is_admin(user):
    return user.is_superuser

def home_view(request):
    """Display the home page."""
    return render(request, 'home.html')

def register_view(request):
    """Handle user registration."""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')  # Redirect to login after registration
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

@login_required
def home_redirect(request):
    """Redirect users to their respective dashboards."""
    if request.user.is_superuser:
        return redirect('admin_dashboard')  # Admin dashboard
    else:
        return redirect('user_dashboard')  # User dashboard

@login_required
@user_passes_test(is_admin)
def admin_dashboard_view(request):
    """Admin dashboard view."""
    datasets = Dataset.objects.all()  # Admin can see all datasets
    user_activity = UserActivityLog.objects.order_by('-timestamp')[:10]  # Last 10 actions
    return render(request, 'admin_dashboard.html', {
        'datasets': datasets,
        'user_activity': user_activity,
    })


@login_required
def user_dashboard_view(request):
    """User dashboard view to handle dataset upload and display analysis."""
    datasets = Dataset.objects.filter(uploaded_by=request.user)  # Fetch datasets uploaded by the user
    analysis_results = []  # Placeholder for analysis results

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Save the uploaded file to the Dataset model
            dataset = Dataset.objects.create(
                name=file.name,
                file=file,
                uploaded_by=request.user
            )
            dataset.save()

            # Log the user action
            UserActivityLog.objects.create(
                user=request.user,
                action=f"Uploaded dataset: {file.name}"
            )

            # Analyze the uploaded file (optional: add your analysis logic here)
            df = pd.read_csv(dataset.file.path)  # Load the uploaded CSV file
            analysis, _ = analyze_dataset(df)
            analysis_results.append(analysis)

        except Exception as e:
            analysis_results.append(f"Error analyzing the dataset: {e}")

    return render(request, 'user_dashboard.html', {
        'datasets': datasets,
        'analysis_results': analysis_results,
    })

def analyze_dataset(df):
    """Analyze the dataset and extract relevant metadata."""
    analysis = {
        'data_shape': df.shape,
        'null_values': df.isnull().values.any(),
    }

    # Detect the target column for fraud
    target_column = None
    if 'Class' in df.columns:
        target_column = 'Class'
    elif 'isFraud' in df.columns:
        target_column = 'isFraud'
    elif 'FraudIndicator' in df.columns:
        target_column = 'FraudIndicator'

    if target_column:
        total_transactions = len(df)
        fraud_transactions = df[target_column].value_counts().get(1, 0)
        normal_transactions = df[target_column].value_counts().get(0, 0)

        analysis.update({
            'unique_target_values': df[target_column].unique(),
            'percentage_non_fraudulent': f"{(normal_transactions / total_transactions) * 100:.3f}%",
            'percentage_fraudulent': f"{(fraud_transactions / total_transactions) * 100:.3f}%",
            'total_fraud_transactions': fraud_transactions,
            'total_normal_transactions': normal_transactions,
        })
    else:
        analysis.update({
            'unique_target_values': "N/A",
            'percentage_non_fraudulent': "N/A",
            'percentage_fraudulent': "N/A",
            'total_fraud_transactions': "N/A",
            'total_normal_transactions': "N/A",
        })

    return analysis, target_column


@login_required
def fraud_detection_view(request):
    result = None
    scatter_url = None
    dataset_analysis = None

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            # Load the dataset
            df = pd.read_csv(file)

            # Analyze the dataset
            dataset_analysis = {
                'Data Shape': df.shape,
                'Null Values': df.isnull().values.any()
            }
            numeric_df = df.select_dtypes(include='number')

            # Ensure numeric data exists
            if numeric_df.empty:
                raise ValueError("The dataset must contain numeric columns for analysis.")

            # Load or train the Isolation Forest model
            retrain_model = False
            if os.path.exists(ISOLATION_MODEL_PATH):
                isolation_model = joblib.load(ISOLATION_MODEL_PATH)
                print("Isolation Forest model loaded successfully.")

                # Check if features match; retrain if not
                if set(isolation_model.feature_names_in_) != set(numeric_df.columns):
                    print("Feature mismatch detected. Retraining model.")
                    retrain_model = True
            else:
                retrain_model = True

            # Train the model if needed
            if retrain_model:
                isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
                isolation_model.fit(numeric_df)
                joblib.dump(isolation_model, ISOLATION_MODEL_PATH)
                print("Isolation Forest model trained and saved.")

            # Perform anomaly detection
            df['anomaly'] = isolation_model.predict(numeric_df)
            anomalies = df[df['anomaly'] == -1]
            dataset_analysis.update({
                'Total Transactions': len(df),
                'Anomalies Detected': len(anomalies),
            })

            # Create scatter plot
            feature_x = 'Time' if 'Time' in numeric_df.columns else numeric_df.columns[0]
            feature_y = 'Amount' if 'Amount' in numeric_df.columns else numeric_df.columns[1]

            fig, ax = plt.subplots()
            normal_data = df[df['anomaly'] != -1]
            anomalous_data = df[df['anomaly'] == -1]
            ax.scatter(normal_data[feature_x], normal_data[feature_y], color='blue', label='Normal', alpha=0.6)
            ax.scatter(anomalous_data[feature_x], anomalous_data[feature_y], color='red', label='Anomaly', alpha=0.6)
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            ax.set_title(f'Scatter Plot: {feature_y} vs {feature_x}')
            ax.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

        except Exception as e:
            result = f"Error processing file: {e}"

    return render(request, 'fraud_detection.html', {
        'dataset_analysis': dataset_analysis,
        'scatter_url': scatter_url,
        'result': result,
    })

@login_required
@user_passes_test(is_admin)

def export_anomalies_to_csv(request):
    anomalies = request.session.get('anomalies_result', [])
    if not anomalies:
        return HttpResponse("No anomalies to export.", content_type="text/plain")

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="anomalies.csv"'

    writer = csv.writer(response)
    writer.writerow(anomalies[0].keys())
    for anomaly in anomalies:
        writer.writerow(anomaly.values())

    # Log the user action
    UserActivityLog.objects.create(
        user=request.user,
        action="Exported anomalies to CSV"
    )
    return response