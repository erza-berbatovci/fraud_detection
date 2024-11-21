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
        # Handle file upload
        if 'file' in request.FILES:
            file = request.FILES['file']
            try:
                # Save the uploaded file
                dataset = Dataset.objects.create(
                    name=file.name,
                    file=file,
                    uploaded_by=request.user
                )
                dataset.save()

                # Debugging logs
                print("Uploaded dataset path (file saved):", dataset.file.path)
                print("File exists after upload:", os.path.exists(dataset.file.path))  # Check file existence

                # Ensure the file is in the correct directory
                destination_path = os.path.join('media', 'datasets', file.name)
                if not os.path.exists(destination_path):
                    shutil.move(dataset.file.path, destination_path)
                    print("File manually moved to:", destination_path)

                messages.success(request, f"Dataset '{file.name}' uploaded successfully.")
            except Exception as e:
                print(f"Error saving dataset: {e}")  # Log the error
                messages.error(request, "Failed to upload dataset. Please try again.")
                return redirect('user_dashboard')

        # Handle dataset analysis
        elif 'dataset_id' in request.POST:
            dataset_id = request.POST['dataset_id']
            try:
                dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
                print("Dataset Path from Database:", dataset.file.path)  # Debugging log

                # Verify if the file exists
                if not os.path.exists(dataset.file.path):
                    print(f"File not found: {dataset.file.path}")  # Debugging log
                    raise FileNotFoundError(f"The file at {dataset.file.path} does not exist.")

                # Load and analyze the dataset
                df = pd.read_csv(dataset.file.path)
                print("Dataset loaded successfully. Columns:", df.columns)  # Debugging log

                # Perform analysis
                analysis_results = {
                    'data_shape': df.shape,
                    'null_values': df.isnull().sum().to_dict(),
                }
                print("Analysis Results:", analysis_results)  # Debugging log

                # Scatter plot generation
                feature_x = 'Time' if 'Time' in df.columns else df.columns[0]
                feature_y = 'Amount' if 'Amount' in df.columns else df.columns[1]

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
            except FileNotFoundError as e:
                print(f"Error: {e}")
                messages.error(request, "The dataset file is missing or has been deleted.")
                return redirect('user_dashboard')
            except Exception as e:
                print(f"Error analyzing dataset: {e}")
                messages.error(request, "Failed to analyze dataset. Please try again.")
                return redirect('user_dashboard')

        # Handle dataset deletion
        elif 'delete_dataset' in request.POST:
            dataset_id = request.POST['delete_dataset']
            try:
                dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)
                print("Dataset to delete:", dataset.file.path)  # Debugging log
                dataset.delete()
                messages.success(request, f"Dataset '{dataset.name}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting dataset: {e}")
                messages.error(request, "Failed to delete dataset. Please try again.")
                return redirect('user_dashboard')

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

@login_required
def fraud_detection_view(request):
    """Handle fraud detection for a specific dataset."""
    scatter_url = None
    dataset_analysis = None
    anomalies_count = 0

    # Get the dataset ID from the query parameter
    dataset_id = request.GET.get('dataset_id')
    dataset = None

    if dataset_id:
        dataset = get_object_or_404(Dataset, id=dataset_id, uploaded_by=request.user)

    if request.method == 'POST' or dataset:
        try:
            # Load the dataset
            if dataset:
                df = pd.read_csv(dataset.file.path)
            else:
                file = request.FILES['file']
                df = pd.read_csv(file)

            # Analyze the dataset
            numeric_df = df.select_dtypes(include='number').copy()
            if numeric_df.empty:
                raise ValueError("The dataset must contain numeric columns for anomaly detection.")

            # Anomaly detection logic
            isolation_model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
            isolation_model.fit(numeric_df)
            df['anomaly'] = isolation_model.predict(numeric_df)
            df.to_csv(dataset.file.path, index=False) 
            anomalies = df[df['anomaly'] == -1]
            normal_data = df[df['anomaly'] != -1]
            anomalies_count = len(anomalies)

            print(df.columns)


            # Dataset analysis for the table
            fraud_count = len(anomalies)
            normal_count = len(normal_data)
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

            # Create a single figure with scatter plot and pie chart
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

            # Save the combined plot
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            scatter_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()

        except Exception as e:
            print(f"Error during processing: {e}")
            messages.error(request, "Error processing the dataset.")
            return redirect('user_dashboard')

    return render(request, 'fraud_detection.html', {
        'scatter_url': scatter_url,
        'dataset_analysis': dataset_analysis,
        'anomalies_count': anomalies_count,
        'dataset_id': dataset_id,
    })

def export_anomalies_excel(request, dataset_id):
    """Export anomalies as an Excel file."""
    dataset = get_object_or_404(Dataset, id=dataset_id)

    try:
        # Load the dataset
        df = pd.read_csv(dataset.file.path)
        print(f"Columns in dataset: {df.columns}")  # Debug: Print the column names
        print(df.head())  # Debug: Print the first few rows of the dataframe

        # Ensure the 'anomaly' column exists
        if 'anomaly' not in df.columns:
            return HttpResponse("No anomaly data available for export.", content_type="text/plain")

        # Filter anomalies
        anomalies = df[df['anomaly'] == -1]

        # Check if there are any anomalies
        if anomalies.empty:
            return HttpResponse("No anomalies detected in the dataset.", content_type="text/plain")

        # Create Excel response
        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="anomalies_{dataset.name}.xlsx"'

        # Write to Excel
        with pd.ExcelWriter(response, engine='openpyxl') as writer:
            anomalies.to_excel(writer, index=False, sheet_name='Anomalies')

        return response
    except Exception as e:
        print(f"Error: {e}")  # Debug: Print the error
        return HttpResponse(f"Error exporting anomalies: {e}", content_type="text/plain")
    
