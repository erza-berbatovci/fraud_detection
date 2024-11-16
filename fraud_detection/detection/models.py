from django.db import models
from django.contrib.auth.models import User

class FraudTransaction(models.Model):
    transaction_id = models.IntegerField(unique=True)
    amount = models.FloatField()
    time = models.FloatField()  # Përdorim float për të ruajtur kohën në formatin epoch
    location = models.IntegerField()  # Përdorim integer për të ruajtur kodin e vendndodhjes

    def __str__(self):
        return f"Transaction {self.transaction_id} - Amount: {self.amount}"



class Dataset(models.Model):
    """
    Model to store information about datasets uploaded by users.
    """
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class UserActivityLog(models.Model):
    """
    Model to log user actions such as uploading files or exporting data.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.action} at {self.timestamp}"
