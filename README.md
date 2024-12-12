# Fraud Detection in Financial Transactions using AI/ML Algorithms

This application is developed as part of a thesis project to detect fraud attempts in financial transactions using advanced AI/ML algorithms. It leverages anomaly detection techniques to identify irregularities in transaction datasets, providing valuable insights and tools for fraud prevention.

## Features

- **User Authentication**: Secure login and registration system with role-based access (Admin and User).  
- **Dataset Upload**: Users can upload CSV files containing financial transaction data for analysis.  
- **Anomaly Detection**: Implements the Isolation Forest algorithm to detect suspicious transactions.  
- **Result Visualization**: Provides graphical representations of identified anomalies, including scatter plots, pie charts, histograms, and box plots.  
- **Admin Dashboard**: Allows administrators to monitor user activities, analyze datasets, and manage uploaded files.  
- **Export Functionality**: Enables downloading of identified anomalies as an Excel report for further processing.  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-folder

## Application Architecture

The application is built using Django, a high-level Python web framework, and leverages several Machine Learning libraries for data analysis and visualization. The architecture consists of the following key components:

### 1. **Backend**
The backend is developed using Django and includes the following functionalities:
- **User Authentication:** Manages login, registration, and role-based access (Admin/User).
- **Dataset Management:** Handles the upload, analysis, and deletion of datasets.
- **Anomaly Detection:** Implements the Isolation Forest algorithm for detecting outliers in financial transactions.

### 2. **Frontend**
The frontend is built using Django templates and includes:
- **User Dashboard:** A user-friendly interface for uploading datasets, viewing analysis results, and exporting reports.
- **Admin Dashboard:** A dedicated interface for administrators to monitor user activities and manage datasets.

### 3. **Database**
The application uses SQLite as the default database (can be switched to PostgreSQL or MySQL for production). The database stores:
- User details and roles.
- Uploaded datasets and their metadata.
- Activity logs for tracking user actions.

## Example Usage Flow

1. **Login/Registration**  
   Users register or log in using their email and password. Admin users have advanced permissions for monitoring and dataset management.

2. **Upload Dataset**  
   Navigate to the dashboard and upload a CSV file containing transaction data. The application automatically stores the dataset in the database.

3. **Analyze Dataset**  
   Select a dataset for analysis. The application uses the Isolation Forest algorithm to detect anomalies and categorize transactions as normal or suspicious.

4. **Visualize Results**  
   View the analysis results in the form of scatter plots, pie charts, histograms, and box plots. These visualizations provide insights into the data and detected anomalies.

5. **Export Anomalies**  
   Download the anomalies as an Excel report for further review or documentation.

## Key Algorithms and Techniques

- **Isolation Forest:** A machine learning algorithm used for anomaly detection. It isolates anomalies based on the distance from other data points in the feature space.
- **Data Visualization:** Includes scatter plots, pie charts, and histograms to display the distribution and analysis results.
- **Role-Based Access Control (RBAC):** Ensures that admin users have exclusive access to monitoring and management functionalities, while regular users can only upload and analyze their datasets.


## Future Enhancements

1. **Integration with APIs:**  
   Add support for real-time transaction data analysis by integrating with financial APIs.

2. **Support for Advanced ML Models:**  
   Include additional algorithms like Random Forests, XGBoost, and neural networks for more robust anomaly detection.

3. **Enhanced Reporting:**  
   Expand reporting capabilities with customizable reports and additional data insights.

4. **Scalability:**  
   Upgrade the application to handle larger datasets and deploy it on cloud platforms for improved scalability.



