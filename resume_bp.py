from flask import Blueprint, request, jsonify
import uuid, re
from semantic import get_embeddings

# Dicionário global simples para armazenar dados do currículo por resume_id
resumes = {}

resume_bp = Blueprint('resume_bp', __name__, url_prefix='/resume')

def extract_keywords(text):
    """
    Extrai palavras-chave básicas do texto (palavras alfabéticas de tamanho > 4).
    """
    tokens = re.findall(r'\w+', text.lower())
    keywords = [w for w in tokens if w.isalpha() and len(w) > 4]
    return sorted(set(keywords))

@resume_bp.route('/', methods=['POST'])
def upload_resume():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Dados do currículo são necessários (campo 'text')."}), 400
    resume_text = data['text']
    # Extrai palavras-chave simples
    keywords = extract_keywords(resume_text)
    # Gera embedding do currículo via Hugging Face
    try:
        resume_emb = get_embeddings([resume_text])[0]
    except Exception as e:
        return jsonify({"error": f"Falha ao gerar embedding do currículo: {e}"}), 500
    # Gera um ID único para este currículo e armazena no dicionário
    resume_id = str(uuid.uuid4())
    resumes[resume_id] = {
        'keywords': keywords,
        'embedding': resume_emb
    }
    return jsonify({"resume_id": resume_id, "keywords": keywords}), 200
