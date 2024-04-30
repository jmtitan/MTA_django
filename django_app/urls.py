from django.urls import path

app_name = 'MTA_app'
from .views import *

urlpatterns = [
    path('attr/', attr, name='attr'),
    path('stat/', stat, name='stat'),
    path('SVcal/', SVcal, name='SVcal'),
    path('wait/', wait, name='wait'),
    path('res_process/', res_process, name='res_process'),

    path('test/', test, name='test'),
]