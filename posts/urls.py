
from django.contrib import admin
from django.urls import path
from django.conf.urls import url,include,static
from posts import views
from django.conf import settings

app_name="posts"
urlpatterns = [
    path("", views.post_home, name='post_home'),
    path("create/", views.post_create, name='post_create'),
    path("<int:id>/", views.post_detail,name='detail'),
    path("<int:id>/edit", views.post_update,name='update'),
    path("<int:id>/delete", views.post_delete,name='delete'),
    path("<int:id>/result", views.test, name='result'),

]
