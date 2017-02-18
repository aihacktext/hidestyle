from django import forms

class NewDocForm(forms.Form):
    new_doc = forms.CharField(widget=forms.TextInput(attrs={'size': '500'}))
