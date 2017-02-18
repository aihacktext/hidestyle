from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /polls/
    url(r'^$', views.index, name='index'),
    # ex: /polls/5/
    url(r'change/$', views.change, name='change'),

    url(r'output/$', views.output, name='output'),
]
