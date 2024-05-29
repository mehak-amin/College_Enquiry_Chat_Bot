from college import app, bcrypt
from flask import render_template, redirect, url_for, flash, request, jsonify
from college.models import  User
from college.forms import LoginForm , RegisterForm
from college import db
from flask_login import login_user, logout_user, login_required, current_user
from college.chat import get_response

def login_check():
    return current_user.is_authenticated if current_user else False

@app.route("/") 
@app.route("/home")
def index_page():
    return render_template('index.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text, login_check())
    message = {"answer": response}  
    return jsonify(message)


@app.route("/about")
@login_required
def about_page():
    return render_template('about.html')

@app.route("/contact")
@login_required
def contact_page():
    return render_template('contact.html')

@app.route("/team")
@login_required
def team_page():
    return render_template('team.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password1.data).decode('utf-8')
        user_to_create = User(username=form.username.data,
                              email_address=form.email_address.data,
                              
                              password_hash=hashed_password)
        db.session.add(user_to_create)
        db.session.commit()

        flash(f"Account created successfully! You are now logged in as {user_to_create.username}", category='success')
        login_user(user_to_create)  
        return redirect(url_for('about_page'))

    if form.errors:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'Error in {field.capitalize()}: {error}', category='danger')

    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()

    if form.validate_on_submit(): 
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('about_page'))
        else:
            flash('Username and password do not match! Please try again', category='danger')

    return render_template('login.html', form=form)
