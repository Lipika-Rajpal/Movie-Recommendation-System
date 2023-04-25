from flask import Blueprint, render_template, request, redirect, url_for
from main import get_movies, add_user, add_rating

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@views.route('/registered_user', methods = ['GET', 'POST'])
def reg_page():
      return render_template('registered_user.html')
        
@views.route('/new_user', methods = ['GET', 'POST'])
def reg_page_new():
      return render_template('new_user.html')

@views.route('/submit_new_user', methods = ['GET', 'POST'])
def get_data_from_html_new():
      name = request.form['name']
      age = request.form['age']
      gender = request.form['gender']
      occupation = request.form['occupation']
      zip_code = request.form['zip']
      uid = int(add_user(name, age, gender, occupation, zip_code))
      a = get_movies(uid, 'anything', True)
      add_rating(uid, a)
      return render_template('get_ratings.html', movie_list=a, lost= f'{uid}')

@views.route('/send_data', methods = ['GET', 'POST'])
def get_data_from_html():
        movie_name = request.form['movie_name']
        uid = request.form['uid']
        a = get_movies(int(uid), movie_name, False)
        print(type(a))
        return render_template('display.html', movie_list=a)
        
