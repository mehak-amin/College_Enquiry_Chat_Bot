from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///college.db'
app.config['SECRET_KEY'] = '82d4fa37f140fe4196d46e9e'

db = SQLAlchemy(app)

with app.app_context():
    db.create_all()
    

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login_page"
login_manager.login_message_category = "info"

from college import routes