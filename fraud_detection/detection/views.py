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
from django.shortcuts import get_object_or_404
from django.contrib import messages

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
    """User dashboard view to handle dataset upload, deletion, and analysis."""
    datasets = Dataset.objects.filter(uploaded_by=request.user)  # Fetch user-specific datasets
    analysis_results = None
    scatter_url = None

    if request.method == 'POST':
        if 'file' in request.FILES:  # File upload logic
            file = request.FILES['file']
            dataset = Dataset.objects.create(
                name=file.name,
                file=file,
                uploaded_by=request.user
            )
            dataset.save()
            messages.success(request, f"Dataset '{file.name}' uploaded successfully.")
        elif 'dataset_id' in request.POST:  # Fraud analysis logic
            dataset_id = request.POST['dataset_id']
            dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
            try:
                # Load dataset and perform analysis
                df = pd.read_csv(dataset.file.path)
                analysis_results, _ = analyze_dataset(df)

                # Scatter plot generation
                feature_x = 'Time' if 'Time' in df.columns else df.columns[0]
                feature_y = 'Amount' if 'Amount' in df.columns else df.columns[1]

                fig, ax = plt.subplots()
                ax.scatter(df[feature_x], df[feature_y], color='blue', alpha=0.6)
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                ax.set_title(f"Scatter Plot: {feature_y} vs {feature_x}")

                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)

            except Exception as e:
                messages.error(request, f"Error analyzing dataset: {e}")
        elif 'delete_dataset' in request.POST:  # Dataset deletion logic
            dataset_id = request.POST['delete_dataset']
            dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
            dataset.delete()
            messages.success(request, f"Dataset '{dataset.name}' deleted successfully.")

    return render(request, 'user_dashboard.html', {
        'datasets': datasets,
        'analysis_results': analysis_results,
        'scatter_url': scatter_url,
    })

@login_required
def delete_dataset(request, dataset_id):
    """Delete a specific dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
    dataset.delete()
    messages.success(request, f"Dataset '{dataset.name}' deleted successfully.")
    return redirect('user_dashboard')


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
    """Handle fraud detection for a specific dataset."""
    result = None
    scatter_url = None
    dataset_analysis = None
    classification_report_str = None
    accuracy = None
    anomalies_count = 0
    is_admin_user = request.user.is_superuser  # Check if the user is an admin

    # Get the dataset ID from the query parameter
    dataset_id = request.GET.get('dataset_id')
    dataset = None

    if dataset_id:
        # Fetch the dataset from the database
        dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)

    if request.method == 'POST' or dataset:
        try:
            # Load the dataset from the file if available
            if dataset:
                df = pd.read_csv(dataset.file.path)
            else:
                # If the user uploads a new file
                file = request.FILES['file']
                df = pd.read_csv(file)

            # Analyze the dataset
            dataset_analysis, target_column = analyze_dataset(df)
            numeric_df = df.select_dtypes(include='number').copy()

            if numeric_df.empty:
                raise ValueError("The dataset must contain numeric columns for anomaly detection.")

            # Load or train the Isolation Forest model
            retrain_model = False
            if os.path.exists(ISOLATION_MODEL_PATH):
                isolation_model = joblib.load(ISOLATION_MODEL_PATH)
                print("Isolation Forest model loaded successfully.")

                # Check if the features match; retrain if they don't
                if set(isolation_model.feature_names_in_) != set(numeric_df.columns):
                    print("Feature mismatch detected. Retraining Isolation Forest.")
                    retrain_model = True
            else:
                print("No existing model found. Training a new Isolation Forest model.")
                retrain_model = True

            # Train the model if necessary
            if retrain_model:
                isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
                isolation_model.fit(numeric_df)
                joblib.dump(isolation_model, ISOLATION_MODEL_PATH)
                print("Isolation Forest model trained and saved.")
            else:
                print("Using pre-trained Isolation Forest model.")

            # Anomaly detection
            df['anomaly'] = isolation_model.predict(numeric_df)
            anomalies = df[df['anomaly'] == -1]
            anomalies_count = len(anomalies)
            result = anomalies.to_dict(orient='records')
            request.session['anomalies_result'] = result

            # Scatter plot generation
            feature_x = 'Time' if 'Time' in numeric_df.columns else numeric_df.columns[0]
            feature_y = 'Amount' if 'Amount' in numeric_df.columns else numeric_df.columns[1]

            fig, ax = plt.subplots()
            normal_data = df[df['anomaly'] != -1]
            anomalous_data = df[df['anomaly'] == -1]
            ax.scatter(normal_data[feature_x], normal_data[feature_y], color='blue', label='Normal', alpha=0.6)
            ax.scatter(anomalous_data[feature_x], anomalous_data[feature_y], color='red', label='Anomaly', alpha=0.6)
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            ax.set_title(f'Scatter Plot: {feature_y} vs {feature_x} (Anomalies Highlighted)')
            ax.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            # Random Forest model (if target column exists) for admins only
            if target_column and is_admin_user:
                X = numeric_df
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                y_pred = rf_model.predict(X_test)
                classification_report_str = classification_report(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)

        except Exception as e:
            result = f"Error during processing: {e}"

    return render(request, 'fraud_detection.html', {
        'form': TransactionForm(),
        'result': result,
        'scatter_url': scatter_url,
        'dataset_analysis': dataset_analysis,
        'classification_report': classification_report_str if is_admin_user else None,
        'accuracy': accuracy if is_admin_user else None,
        'anomalies_count': anomalies_count,
        'is_admin_user': is_admin_user,  # Pass admin status to the template
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