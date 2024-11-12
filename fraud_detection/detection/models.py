from django.db import models

class FraudTransaction(models.Model):
    transaction_id = models.IntegerField(unique=True)
    amount = models.FloatField()
    time = models.FloatField()  # Përdorim float për të ruajtur kohën në formatin epoch
    location = models.IntegerField()  # Përdorim integer për të ruajtur kodin e vendndodhjes

    def __str__(self):
        return f"Transaction {self.transaction_id} - Amount: {self.amount}"
