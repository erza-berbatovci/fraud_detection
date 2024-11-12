from django.contrib import admin
from django.urls import path
from detection import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detection import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('fraud_detection/', views.fraud_detection_view, name='fraud_detection'),
    path('', views.fraud_detection_view, name='home'),
    path('export/csv/', views.export_to_csv, name='export_to_csv'),
    path('export/excel/', views.export_to_excel, name='export_to_excel'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
