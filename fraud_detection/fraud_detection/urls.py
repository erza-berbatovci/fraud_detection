from django.urls import path
from detection import views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_view, name='home'),  # Home page
    path('fraud_detection/', views.fraud_detection_view, name='fraud_detection'),  # Fraud detection
    path('register/', views.register_view, name='register'),  # Registration page
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),  # Login page
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),  # Logout
    path('redirect/', views.home_redirect, name='redirect'),  # Redirect users after login
    path('admin_dashboard/', views.admin_dashboard_view, name='admin_dashboard'),  # Admin dashboard
    path('user_dashboard/', views.user_dashboard_view, name='user_dashboard'),  # User dashboard
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),

] 

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
