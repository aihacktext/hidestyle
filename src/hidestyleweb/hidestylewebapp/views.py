from django.shortcuts import render

from django.shortcuts import HttpResponse, HttpResponseRedirect, render
from django.template import loader
from django.core.exceptions import *

from .forms import NewDocForm

# Create your views here.
def index(request):
    return HttpResponse("hello world")

def change(request):
    if request.method == 'POST':
        # create a form instance and populate it with the data from the request
        form = NewDocForm(request.POST)

        # if no errors, do your thing
        if form.is_valid():

            # access the data from the form
            text = form.cleaned_data[new_doc]

            # placeholder processing
            processed_text = ''.join([a for a in reversed(text)])

            # Let other views use this processed data
            request.session['processed_text'] = processed_text

            return HttpResponseRedirect('/output/')


    else:
        form = NewDocForm()

    return render(request, "templates/form.html", {"form": form})


def output(request):

    processed_text = request.session.get('processed_text')
    return render(request, "templates/output.html")
