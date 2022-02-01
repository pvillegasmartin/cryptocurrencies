from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder='../templates')
api = Api(app)
app.config.from_object("api.config.Config")
db = SQLAlchemy(app)



class Striver(db.Model):
    __tablename__ = 'strivers'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    email = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(64), unique=False, nullable=True)

    def __init__(self,email, name=""):
        self.email = email
        self.name = name

class Striver_api(Resource):

    def get(self):
        args_parser = reqparse.RequestParser()
        args_parser.add_argument('email', type=str)

        args = args_parser.parse_args()
        email = args['email']

        try:
            striver_info = db.session().query(Striver).filter_by(email=email).first()
            return {'email':striver_info.email, 'name':striver_info.name}

        except:
            return {'ERROR': "Couldn't find email"}

        return {'email':email}

    def post(self):
        args_parser = reqparse.RequestParser()
        args_parser.add_argument('email', type=str)
        args_parser.add_argument('name', type=str)

        args = args_parser.parse_args()
        email = args['email']
        name = args['name']

        print(name)
        print(email)

        try:
            db.session.add(Striver(email=email, name=name))
            db.session.commit()
            return {'email':email, 'name':name}

        except:
            return {'ERROR': "Couldn't insert email"}

api.add_resource(Striver_api, '/striver')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prueba')
def prueba():
    return "Hello world"
