from django.urls import path
from my_app.views import login, video_feed, index
from . import views

urlpatterns = [
    path('login/', login, name='login'),
    path('video_feed/', video_feed, name='video_feed'),
    path('register/', views.register, name='register'),
     path('', index, name='home'),
]