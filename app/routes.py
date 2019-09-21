from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    return render_template('post_.html', title='Home')