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
from openpyxl import Workbook
from .models import UserActivityLog
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import IsolationForest
import numpy as np

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

def is_admin(user):
    """Check if the user is an admin."""
    return user.is_staff

@login_required
@user_passes_test(is_admin)
def admin_dashboard_view(request):
    """Admin dashboard view to handle dataset upload, deletion, and analysis."""
    if not request.user.is_staff:  # Ensure only admins can access this view
        messages.error(request, "You do not have permission to access this page.")
        return redirect('user_dashboard')

    datasets = Dataset.objects.all()  # Admins can see all datasets
    user_activity_logs = UserActivityLog.objects.order_by('-timestamp')[:10]  # Last 10 activities
    scatter_url = None
    dataset_analysis = None

    if request.method == 'POST':
        try:
            # Handle dataset upload
            if 'file' in request.FILES:
                file = request.FILES['file']
                dataset = Dataset.objects.create(
                    name=file.name,
                    file=file,
                    uploaded_by=request.user  # Admin is marked as the uploader
                )
                dataset.save()

                # Log the action
                UserActivityLog.objects.create(
                    user=request.user,
                    action=f"Uploaded a dataset: {dataset.name}"
                )
                messages.success(request, f"Dataset '{file.name}' uploaded successfully.")

            # Handle dataset analysis
            elif 'dataset_id' in request.POST:
                dataset_id = request.POST['dataset_id']
                dataset = get_object_or_404(Dataset, id=dataset_id)

                # Load and analyze the dataset
                df = pd.read_csv(dataset.file.path)
                numeric_df = df.select_dtypes(include='number')
                if numeric_df.empty:
                    raise ValueError("The dataset must contain numeric columns for analysis.")

                # Perform anomaly detection
                isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
                isolation_model.fit(numeric_df)
                df['anomaly'] = isolation_model.predict(numeric_df)
                anomalies = df[df['anomaly'] == -1]
                normal_data = df[df['anomaly'] != -1]
                fraud_count = len(anomalies)
                normal_count = len(normal_data)

                # Log the action
                UserActivityLog.objects.create(
                    user=request.user,
                    action=f"Analyzed dataset: {dataset.name}"
                )

                # Dataset analysis
                total_transactions = len(df)
                dataset_analysis = {
                    'data_shape': df.shape,
                    'null_values': df.isnull().sum().to_dict(),
                    'percentage_non_fraudulent': f"{(normal_count / total_transactions) * 100:.3f}%",
                    'percentage_fraudulent': f"{(fraud_count / total_transactions) * 100:.3f}%",
                    'total_fraud_transactions': fraud_count,
                    'total_normal_transactions': normal_count,
                }

                # Scatter plot and pie chart generation
                feature_x = 'Time' if 'Time' in numeric_df.columns else numeric_df.columns[0]
                feature_y = 'Amount' if 'Amount' in numeric_df.columns else numeric_df.columns[1]

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                # Scatter Plot
                axs[0].scatter(normal_data[feature_x], normal_data[feature_y], color='blue', label='Normal', alpha=0.6)
                axs[0].scatter(anomalies[feature_x], anomalies[feature_y], color='red', label='Anomaly', alpha=0.6)
                axs[0].set_xlabel(feature_x)
                axs[0].set_ylabel(feature_y)
                axs[0].set_title(f'Scatter Plot: {feature_y} vs {feature_x} (Anomalies Highlighted)')
                axs[0].legend()

                # Pie Chart
                axs[1].pie(
                    [normal_count, fraud_count],
                    labels=['Normal', 'Fraudulent'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['blue', 'red']
                )
                axs[1].set_title('Transaction Distribution')

                # Save the combined plot as a Base64 string
                buffer = BytesIO()
                plt.tight_layout()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                plt.close(fig)

        except FileNotFoundError:
            messages.error(request, "The dataset file is missing or has been deleted.")
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"An unexpected error occurred: {e}")

    return render(request, 'admin_dashboard.html', {
        'datasets': datasets,
        'scatter_url': scatter_url,
        'dataset_analysis': dataset_analysis,
        'user_activity_logs': user_activity_logs,
    })

@login_required
def user_dashboard_view(request):
    """User dashboard view to handle dataset upload, deletion, and analysis."""
    datasets = Dataset.objects.filter(uploaded_by=request.user)  # Fetch user-specific datasets
    analysis_results = None
    scatter_url = None

    if request.method == 'POST':
        try:
            # Handle dataset upload
            if 'file' in request.FILES:
                file = request.FILES['file']
                dataset = Dataset.objects.create(
                    name=file.name,
                    file=file,
                    uploaded_by=request.user
                )
                dataset.save()

                # Log the action
                UserActivityLog.objects.create(
                    user=request.user,
                    action=f"Uploaded a dataset: {dataset.name}"
                )
                messages.success(request, f"Dataset '{file.name}' uploaded successfully.")

            # Handle dataset analysis
            elif 'dataset_id' in request.POST:
                dataset_id = request.POST['dataset_id']
                dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)

                if not os.path.exists(dataset.file.path):
                    raise FileNotFoundError(f"The file at {dataset.file.path} does not exist.")

                df = pd.read_csv(dataset.file.path)
                numeric_df = df.select_dtypes(include='number')

                if numeric_df.empty:
                    raise ValueError("The dataset must contain numeric columns for analysis.")

                # Log the action
                UserActivityLog.objects.create(
                    user=request.user,
                    action=f"Analyzed dataset: {dataset.name}"
                )

                # Perform analysis
                analysis_results = {
                    'data_shape': df.shape,
                    'null_values': df.isnull().sum().to_dict(),
                }

                # Scatter plot generation
                feature_x = 'Time' if 'Time' in numeric_df.columns else numeric_df.columns[0]
                feature_y = 'Amount' if 'Amount' in numeric_df.columns else numeric_df.columns[1]

                fig, ax = plt.subplots()
                ax.scatter(df[feature_x], df[feature_y], color='blue', alpha=0.6)
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                ax.set_title("Scatter Plot")
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)

            # Handle dataset deletion
            elif 'delete_dataset' in request.POST:
                dataset_id = request.POST['delete_dataset']
                dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)

                # Log the action
                UserActivityLog.objects.create(
                    user=request.user,
                    action=f"Deleted a dataset: {dataset.name}"
                )
                dataset.delete()
                messages.success(request, f"Dataset '{dataset.name}' deleted successfully.")

        except FileNotFoundError as e:
            messages.error(request, "The dataset file is missing or has been deleted.")
        except ValueError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, "An unexpected error occurred. Please try again.")

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
    print("Analyzing dataset...")
    print("Columns:", df.columns)
    print("First few rows:\n", df.head())
    
    analysis = {
        'data_shape': df.shape,
        'null_values': df.isnull().values.any(),
    }

    target_column = None
    if 'Class' in df.columns:
        target_column = 'Class'
    elif 'isFraud' in df.columns:
        target_column = 'isFraud'
    elif 'FraudIndicator' in df.columns:
        target_column = 'FraudIndicator'
    
    print("Target column detected:", target_column)
    
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
        print("No target column found.")
        analysis.update({
            'unique_target_values': "N/A",
            'percentage_non_fraudulent': "N/A",
            'percentage_fraudulent': "N/A",
            'total_fraud_transactions': "N/A",
            'total_normal_transactions': "N/A",
        })

    print("Analysis results:", analysis)
    return analysis, target_column





# Custom scorer for unsupervised learning
def unsupervised_scorer(estimator, X):
    predictions = estimator.fit_predict(X)
    anomaly_ratio = (predictions == -1).sum() / len(predictions)
    return anomaly_ratio

custom_scorer = make_scorer(unsupervised_scorer, greater_is_better=True)

@login_required
def fraud_detection_view(request, dataset_id=None):
    scatter_url = None
    dataset_analysis = None
    anomalies_count = 0
    cross_val_scores = None

    if dataset_id is None:
        messages.error(request, "No dataset specified for analysis.")
        return redirect('user_dashboard')

    dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)

    try:
        # Ngarkimi i dataset-it
        df = pd.read_csv(dataset.file.path)
        numeric_df = df.select_dtypes(include='number')

        if numeric_df.empty:
            raise ValueError("The dataset must contain numeric columns for analysis.")

        # Krijimi i modelit Isolation Forest
        isolation_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

        # KFold Cross-Validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in kf.split(numeric_df):
            X_train, X_test = numeric_df.iloc[train_index], numeric_df.iloc[test_index]
            isolation_model.fit(X_train)
            predictions = isolation_model.predict(X_test)
            anomaly_ratio = (predictions == -1).sum() / len(predictions)
            scores.append(anomaly_ratio)

        cross_val_scores = list(scores)  # Konverto në listë për template

        # Handle train-test split for visualization
        X_train, X_test = train_test_split(numeric_df, test_size=0.2, random_state=42)
        isolation_model.fit(X_train)
        X_test['anomaly'] = isolation_model.predict(X_test)
        anomalies = X_test[X_test['anomaly'] == -1]
        normal_data = X_test[X_test['anomaly'] != -1]

        fraud_count = len(anomalies)
        normal_count = len(normal_data)
        anomalies_count = fraud_count

        total_transactions = len(X_test)
        dataset_analysis = {
            'data_shape': X_test.shape,
            'null_values': X_test.isnull().sum().to_dict(),
            'percentage_non_fraudulent': f"{(normal_count / total_transactions) * 100:.3f}%",
            'percentage_fraudulent': f"{(fraud_count / total_transactions) * 100:.3f}%",
            'total_fraud_transactions': fraud_count,
            'total_normal_transactions': normal_count,
        }

        # Scatter plot dhe pie chart
        feature_x = 'Time' if 'Time' in numeric_df.columns else numeric_df.columns[0]
        feature_y = 'Amount' if 'Amount' in numeric_df.columns else numeric_df.columns[1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].scatter(normal_data[feature_x], normal_data[feature_y], color='blue', label='Normal', alpha=0.6)
        axs[0].scatter(anomalies[feature_x], anomalies[feature_y], color='red', label='Anomaly', alpha=0.6)
        axs[0].set_xlabel(feature_x)
        axs[0].set_ylabel(feature_y)
        axs[0].legend()

        axs[1].pie(
            [normal_count, fraud_count],
            labels=['Normal', 'Fraudulent'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['blue', 'red']
        )
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close(fig)

    except Exception as e:
        messages.error(request, f"Error: {e}")
        return redirect('user_dashboard')

    return render(request, 'fraud_detection.html', {
    'scatter_url': scatter_url,
    'dataset_analysis': dataset_analysis,
    'anomalies_count': anomalies_count,
    'cross_val_scores': cross_val_scores,
    'cross_val_mean': np.mean(cross_val_scores) if cross_val_scores else None,
    'dataset_id': dataset_id,  # Kalimi i dataset_id në template
})

def about(request):
    return render(request, 'about.html')

def export_anomalies_excel(request, dataset_id):
    """Export anomalies as an Excel file."""
    dataset = get_object_or_404(Dataset, id=dataset_id)

    try:
        # Load the dataset
        df = pd.read_csv(dataset.file.path)

        # Shto kolonën 'anomaly' nëse mungon
        if 'anomaly' not in df.columns:
            isolation_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            isolation_model.fit(df.select_dtypes(include='number'))
            df['anomaly'] = isolation_model.predict(df.select_dtypes(include='number'))
            df.to_csv(dataset.file.path, index=False)  # Ruaj dataset-in të përditësuar

        # Filtroni anomalitë
        anomalies = df[df['anomaly'] == -1]

        # Kontrolloni nëse ka të dhëna anomalie
        if anomalies.empty:
            return HttpResponse("Nuk ka të dhëna anomalie për eksportim.", content_type="text/plain")

        # Krijoni një Excel file për eksport
        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="anomalies_{dataset.name}.xlsx"'

        with pd.ExcelWriter(response, engine='openpyxl') as writer:
            anomalies.to_excel(writer, index=False, sheet_name='Anomalies')

        return response
    except Exception as e:
        return HttpResponse(f"Gabim gjatë eksportimit të anomalive: {e}", content_type="text/plain")
