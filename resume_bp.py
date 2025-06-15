from flask import Blueprint, request, jsonify
from uuid import uuid4
from keybert import KeyBERT
from semantic import get_embeddings

# Blueprint do currículo
resume_bp = Blueprint('resume_bp', __name__, url_prefix='/resume')

# Dicionário em memória para armazenar os currículos
resumes = {}

# Modelo KeyBERT com modelo local compatível
kw_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')

def extract_keywords(text):
    """
    Extrai palavras-chave contextuais relevantes usando KeyBERT.
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_mmr=True,
        nr_candidates=30,
        diversity=0.7,
    )
    return sorted(set(kw[0].lower() for kw in keywords))

@resume_bp.route('/', methods=['POST'])
def upload_resume():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Dados do currículo são necessários (campo 'text')."}), 400

    resume_text = data['text']

    try:
        # Extrai palavras-chave
        keywords = extract_keywords(resume_text)
        # Gera embedding do currículo localmente
        resume_emb = get_embeddings([resume_text])[0]
    except Exception as e:
        return jsonify({"error": f"Falha ao processar currículo: {e}"}), 500

    resume_id = str(uuid4())
    resumes[resume_id] = {
        'text': resume_text,
        'keywords': keywords,
        'embedding': resume_emb
    }

    return jsonify({"resume_id": resume_id, "keywords": keywords}), 200
