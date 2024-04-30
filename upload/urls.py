from django.urls import path
from . import views

app_name = 'upload'

urlpatterns = [
    # path('test/', views.upload_file, name='find_file'),
    path('', views.upload_file, name='upload_file')
]
