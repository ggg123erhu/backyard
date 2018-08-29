from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from .models import Post
from django.contrib import messages
from .forms import PostForm
from .classification import Image_identify
# request.user.is_authenticated()
import requests
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


def post_home(request):
    query_set_list = Post.objects.all()#.order_by("-timestamp")
    context = {
        # 'title': 'HOME PAGE',
        # 'Post_query': query_set,
    }
    return render(request, 'list_all.html', context)


def post_update(request,id=None):
    instance = get_object_or_404(Post, id=id)
    form = PostForm(request.POST or None,request.FILES or None,instance = instance)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.save()
        messages.success(request, "successfully update")
        return HttpResponseRedirect(instance.get_absolute_url())
    context = {
        "title": "Detail",
        "instance": instance,
        "form":form
    }
    return render(request, "forms.html", context)


def post_create(request):
    form = PostForm(request.POST or None,request.FILES or None)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.save()
        messages.success(request,"successfully create")
        return HttpResponseRedirect(instance.get_absolute_url())
    else:
        messages.error(request,"Not successfully create yet")
    context = {
        "form": form
    }
    return render(request, "forms.html", context)


def post_detail(request, id=None):
    instance = get_object_or_404(Post, id=id)
    context = {
        "title": "Detail",
        "instance": instance
    }
    return render(request, "detail.html", context)


def post_delete(request, id=None):
    instance = get_object_or_404(Post, id=id)
    instance.delete()
    messages.success(request, "successfully delete")
    return redirect('posts:list')


def classify_image(request, id=None):
    instance = get_object_or_404(Post, id=id)
    test_data = "./media"
    str_label=Image_identify.process_test_data(test_data)
    context={
        "instance": instance,
        "str_label": str_label
    }
    return render(request, "result.html", context)


def test(request, id=None):
    IMG_SIZE = 50
    LR = 1e-3
    instance = get_object_or_404(Post, id=id)
    test_data = "./media"
    testing_data = []
    for img in tqdm(os.listdir(test_data)):
        if img != '.DS_Store':
            path = os.path.join(test_data, img)
            img_num = img.split('.')[-1]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)

    test_data = testing_data

    for num, data in enumerate(test_data):
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

        tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model_import = tflearn.DNN(convnet, tensorboard_dir='log')

        model_import.load("./MODEL_NAME", weights_only=False)

        model_out = model_import.predict([data])[0]
        if np.argmax(model_out) == 1:
            str_label = 'disease'
        else:
            str_label = 'healthy'


    context={
        "instance": instance,
        "str_label": str_label
    }
    return render(request, "result.html", context)
