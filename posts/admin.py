from django.contrib import admin
from .models import Post
# Register your models here.


class PostModelAdmin(admin.ModelAdmin):
    list_display = ['title','updated','timestamp']# add column in admin page
    list_display_links = ['updated']
    list_filter = ['updated','timestamp']
    search_fields = ['title']

    class Meta:
        model = Post

admin.site.register(Post,PostModelAdmin)