

from django.urls import path
from .views import index, query, get_docs

urlpatterns = [
    path('', index, name='index'),
    path('query', query, name='query'),
    path('docs', get_docs, name='get_docs'),
]