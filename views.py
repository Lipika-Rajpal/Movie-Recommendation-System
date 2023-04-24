from flask import Blueprint, render_template, request, redirect, url_for
from main import get_movies

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@views.route('/send_data', methods = ['POST'])
def get_data_from_html():
        movie_name = request.form['movie_name']
        uid = request.form['uid']
        a = get_movies(int(uid), movie_name, True)
        print(type(a))
        return render_template('display.html', movie_list=a)
        
