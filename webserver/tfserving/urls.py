from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include

from rest_framework import routers
from rest_framework_swagger.views import get_swagger_view

from tfserving import api, views

app_name = 'tfserving'

router = routers.DefaultRouter()
router.register('members', api.MemberViewSet)

urlpatterns = [
    
    url(r'^api/doc', get_swagger_view(title='Rest API Document')),
    url(r'^api/v1/', include((router.urls, 'tfserving'), namespace='api')),
       
    url('predict/', views.predict_image, name='predict')
]
