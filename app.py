from flask import Flask
from dotenv import load_dotenv
from resume_bp import resume_bp
from jobs_bp import jobs_bp
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
app.register_blueprint(resume_bp)
app.register_blueprint(jobs_bp)
CORS(app)

@app.route('/')
def index():
    return "API do HireSearcher está ativa!"

if __name__ == '__main__':
    app.run(debug=True)
