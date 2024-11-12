from django import forms

class TransactionForm(forms.Form):
    file = forms.FileField(label='Ngarko të dhënat e transaksioneve', required=True)
