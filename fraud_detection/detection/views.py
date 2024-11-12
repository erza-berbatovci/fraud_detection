from django.shortcuts import render
from .forms import TransactionForm
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from .models import FraudTransaction  # Importimi i modelit për DB
import csv
from django.http import HttpResponse
from openpyxl import Workbook

# Funksioni që trajton ngarkimin dhe analizën
def fraud_detection_view(request):
    result = None
    model = None
    anomaly_ids = []  # Lista e ID-ve të transaksioneve anomali

    # Vendos path për modelin e ruajtur
    model_path = os.path.join('fraud_model.pkl')

    # Kontrollo nëse modeli është i ruajtur në disk dhe ngarko atë
    if os.path.exists(model_path):
        model = joblib.load(model_path)  # Ngarko modelin e ruajtur
    else:
        # Trajnon dhe ruan modelin vetëm një herë
        if request.method == 'POST' and request.FILES.get('file'):
            file = request.FILES['file']

            # Kontrollo nëse file është në formatin e pranuar
            if not file.name.endswith('.csv'):
                result = 'Ju lutem ngarkoni një skedar CSV.'
                return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

            # Kontrollo nëse skedari ka të dhëna
            try:
                # Lexo skedarin CSV
                df = pd.read_csv(file)

                if df.empty:
                    result = 'Skedari është i zbrazët.'
                    return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

                # Sigurohu që të dhënat kanë kolonat e nevojshme
                if 'amount' not in df.columns or 'time' not in df.columns or 'location' not in df.columns:
                    result = 'Kolonat e duhura mungojnë në skedar.'
                    return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

                # Konverto kolonën 'time' në datetime dhe pastaj në format numerik (epoch time)
                df['time'] = pd.to_datetime(df['time'])
                df['time'] = df['time'].apply(lambda x: x.timestamp())  # Konverto në sekonda që nga 1970-01-01

                # Konverto kolonën 'location' nga string në numra
                label_encoder = LabelEncoder()
                df['location'] = label_encoder.fit_transform(df['location'])  # Konverto 'NY', 'LA', etj. në numra

                # Trajno modelin dhe ruaj nëse nuk ekziston
                model = IsolationForest(n_estimators=100, contamination=0.1)
                model.fit(df[['amount', 'time', 'location']])

                # Ruaj modelin në disk për përdorim të mëvonshëm
                joblib.dump(model, model_path)

            except Exception as e:
                result = f'Ka ndodhur një gabim gjatë ngarkimit të skedarit: {e}'
                return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

    # Përdor modelin për të identifikuar anomalitë
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        # Ngarko të dhënat nga skedari i ngarkuar (CSV)
        try:
            df = pd.read_csv(file)

            # Sigurohu që të dhënat kanë kolonat e nevojshme
            if 'amount' not in df.columns or 'time' not in df.columns or 'location' not in df.columns:
                result = 'Kolonat e duhura mungojnë në skedar.'
                return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

            # Konverto kolonën 'time' në datetime dhe pastaj në format numerik (epoch time)
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].apply(lambda x: x.timestamp())  # Konverto në sekonda që nga 1970-01-01

            # Konverto kolonën 'location' nga string në numra
            label_encoder = LabelEncoder()
            df['location'] = label_encoder.fit_transform(df['location'])  # Konverto 'NY', 'LA', etj. në numra

            # Përdor modelin për të identifikuar anomalitë
            df['anomaly'] = model.predict(df[['amount', 'time', 'location']])

            # Gjej transaksionet që janë anomali (të etiketuar me -1)
            anomalies = df[df['anomaly'] == -1]  # Merr të gjitha rreshtat që janë anomali

            print("Anomalies found:", anomalies)  # Printo anomalitë për debugging

            # Ruaj anomalitë në bazën e të dhënave
            for index, row in anomalies.iterrows():
                # Kontrollo nëse 'transaction_id' ekziston në bazën e të dhënave
                if not FraudTransaction.objects.filter(transaction_id=row['transaction_id']).exists():
                    FraudTransaction.objects.create(
                        transaction_id=row['transaction_id'],
                        amount=row['amount'],
                        time=row['time'],
                        location=row['location']
                    )
                else:
                    print(f"Transaction ID {row['transaction_id']} already exists.")  # Debugging

            # Dërgo të dhënat e plota për anomalitë në template
            result = anomalies[['transaction_id', 'amount', 'time', 'location']].to_dict(orient='records')

            print("Result data to be rendered:", result)  # Printo rezultatin që do të dërgohet në template

        except Exception as e:
            result = f'Ka ndodhur një gabim gjatë ngarkimit të skedarit: {e}'
            return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})

    return render(request, 'fraud_detection.html', {'form': TransactionForm(), 'result': result})
def export_to_csv(request):
    # Merrni anomalitë nga databaza
    anomalies = FraudTransaction.objects.all()

    # Krijoni një përgjigje me përmbajtje CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="anomalies.csv"'

    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Amount', 'Time', 'Location'])

    for anomaly in anomalies:
        writer.writerow([anomaly.transaction_id, anomaly.amount, anomaly.time, anomaly.location])

    return response
def export_to_excel(request):
    anomalies = FraudTransaction.objects.all()

    # Krijoni një workbook të ri
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