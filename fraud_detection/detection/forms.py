from django import forms
from .models import Dataset

class TransactionForm(forms.Form):
    file = forms.FileField(label='Ngarko të dhënat e transaksioneve', required=True)



class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['file']