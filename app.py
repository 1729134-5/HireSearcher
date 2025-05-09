from flask import Flask
from dotenv import load_dotenv
from resume_bp import resume_bp
from jobs_bp import jobs_bp

load_dotenv()  # Carrega variáveis do .env

app = Flask(__name__)
app.register_blueprint(resume_bp)
app.register_blueprint(jobs_bp)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return "API do HireSearcher está ativa!"
