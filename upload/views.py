from django.http import HttpResponse
from django.shortcuts import redirect
import os
from .forms import FileForm
from django.shortcuts import render
from django.template.response import TemplateResponse

def upload_choice(request):
    return TemplateResponse(request, 'upload/upload.html')

def upload_file(request):
    if request.method == 'GET':
        form = FileForm()
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return redirect('MTA_app:stat')
            # return HttpResponse('sucess')
        else:
            print(form.errors.get_json_data())
            return HttpResponse('fail')
    return render(request, 'upload/upload.html', {'form': form})

def handle_uploaded_file(f):
    file_name = 'origin.zip'
    count = 1
    while os.path.isfile('MTA_app/files/' + file_name):
        file_name = file_name.split('.')[0][:6] + str(count) + '.zip'
    with open('MTA_app/files/origin.zip', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def test(request):
    return TemplateResponse(request, 'upload/upload.html')