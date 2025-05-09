from flask import Blueprint, request, jsonify
import os, requests
from dotenv import load_dotenv
from semantic import get_embeddings, cosine_similarity

load_dotenv()  # Carrega CATHO_CLIENT_ID e CATHO_CLIENT_SECRET do .env

CATHO_CLIENT_ID = os.getenv("CATHO_CLIENT_ID")
CATHO_CLIENT_SECRET = os.getenv("CATHO_CLIENT_SECRET")
CATHO_BASE_URL = os.getenv("CATHO_BASE_URL", "https://api.catho.com.br")

jobs_bp = Blueprint('jobs_bp', __name__, url_prefix='/jobs')

# Cache simples: chave=tuple de palavras-chave, valor=lista de vagas retornada
job_cache = {}

def get_catho_token():
    """
    Obtém o token de acesso OAuth2 da Catho usando client credentials.
    """
    token_url = f"{CATHO_BASE_URL}/oauth/token"
    data = {'grant_type': 'client_credentials'}
    try:
        resp = requests.post(token_url, data=data, auth=(CATHO_CLIENT_ID, CATHO_CLIENT_SECRET))
        resp.raise_for_status()
        token = resp.json().get('access_token')
        return token
    except Exception as e:
        raise RuntimeError(f"Erro na autenticação Catho: {e}")

def search_catho_jobs(keywords):
    """
    Busca vagas na API da Catho usando as palavras-chave. Usa cache simples.
    Retorna a lista de vagas (como dicts).
    """
    key = tuple(keywords)
    if key in job_cache:
        return job_cache[key]
    # Monta requisição à API de vagas (exemplo de endpoint)
    search_url = f"{CATHO_BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {get_catho_token()}"}
    params = {"keywords": ",".join(keywords)}
    resp = requests.get(search_url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    # Ajusta de acordo com o formato real da resposta (supondo 'jobs' ou lista)
    jobs = data.get('jobs') if isinstance(data, dict) else data
    if jobs is None:
        jobs = []
    job_cache[key] = jobs
    return jobs

@jobs_bp.route('/', methods=['POST'])
def find_jobs():
    data = request.json
    resume_id = data.get('resume_id') if data else None
    if not resume_id:
        return jsonify({"error": "resume_id é obrigatório."}), 400
    if resume_id not in __import__('resume_bp').resumes:
        return jsonify({"error": "resume_id não encontrado."}), 404
    # Recupera palavras-chave e embedding do currículo armazenados
    from resume_bp import resumes
    keywords = resumes[resume_id]['keywords']
    resume_emb = resumes[resume_id]['embedding']
    if not keywords:
        return jsonify({"jobs": []}), 200

    # Busca vagas na Catho
    try:
        jobs_list = search_catho_jobs(keywords)
    except Exception as e:
        return jsonify({"error": f"Falha ao buscar vagas na Catho: {e}"}), 500

    if not jobs_list:
        return jsonify({"jobs": []}), 200

    # Extrai descrições das vagas (assumindo campo 'description' ou 'descricao')
    descriptions = []
    for job in jobs_list:
        desc = job.get('description') or job.get('descricao') or ""
        descriptions.append(desc)

    # Gera embeddings das descrições
    try:
        job_embeddings = get_embeddings(descriptions)
    except Exception as e:
        return jsonify({"error": f"Falha ao gerar embeddings das vagas: {e}"}), 500

    # Calcula similaridades e seleciona top 3
    scored_jobs = []
    for job, emb in zip(jobs_list, job_embeddings):
        score = cosine_similarity(resume_emb, emb)
        scored_jobs.append((score, job))
    # Ordena por similaridade decrescente
    scored_jobs.sort(key=lambda x: x[0], reverse=True)
    top3 = scored_jobs[:3]

    # Prepara resultado JSON
    result = []
    for score, job in top3:
        title = job.get('title') or job.get('titulo') or ""
        desc = job.get('description') or job.get('descricao') or ""
        result.append({
            "title": title,
            "description": desc,
            "similarity": round(score, 4)
        })
    return jsonify({"status": "success", "jobs": result}), 200
