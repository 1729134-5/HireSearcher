from flask import Blueprint, request, jsonify
import requests
from semantic import get_embeddings, cosine_similarity
from resume_bp import resumes
from html.parser import HTMLParser
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

jobs_bp = Blueprint('jobs_bp', __name__, url_prefix='/jobs')

# Cache em memória para resultados por termo combinado
job_cache = {}

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return ''.join(self.text_parts).strip()

def clean_html(raw_html):
    parser = HTMLTextExtractor()
    parser.feed(raw_html or "")
    text = parser.get_text()
    return text.replace('\n', '<br>')

def fetch_jobs(query):
    if query in job_cache:
        print(f"📦 Cache: {query}")
        return query, job_cache[query]

    print(f"🔍 Buscando: {query}")
    try:
        resp = requests.get("https://remotive.com/api/remote-jobs", params={"search": query})
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
        job_cache[query] = jobs
        return query, jobs
    except Exception as e:
        print(f"❌ Erro para '{query}': {e}")
        return query, []

def search_remotive_jobs(keywords):
    """
    Busca vagas do Remotive com base em combinações de palavras-chave.
    Executa buscas em paralelo e acumula até 5 resultados únicos.
    """
    terms = set()
    for phrase in keywords:
        for word in phrase.lower().split():
            if len(word) > 3:
                terms.add(word)

    terms = list(terms)
    if not terms:
        return []

    seen_ids = set()
    jobs = []

    # Gera combinações de 3, 2 e 1 termo
    queries = []
    for size in range(min(3, len(terms)), 0, -1):
        queries += [" ".join(combo) for combo in combinations(terms, size)]

    # Executa buscas em paralelo
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_jobs, q) for q in queries]
        for future in as_completed(futures):
            query, new_jobs = future.result()
            for job in new_jobs:
                job_id = job.get("id") or job.get("url") or job.get("title") + job.get("company_name", "")
                if job_id not in seen_ids:
                    jobs.append(job)
                    seen_ids.add(job_id)
            if len(jobs) >= 5:
                break

    return jobs

@jobs_bp.route('/', methods=['POST'])
def find_jobs():
    data = request.json
    resume_id = data.get('resume_id') if data else None

    if not resume_id:
        return jsonify({"error": "resume_id é obrigatório."}), 400
    if resume_id not in resumes:
        return jsonify({"error": "resume_id não encontrado."}), 404

    resume_data = resumes[resume_id]
    resume_text = resume_data.get('text')
    resume_embedding = resume_data.get('embedding')
    keywords = resume_data.get('keywords')

    if not resume_text or not resume_embedding:
        return jsonify({"error": "Currículo inválido."}), 500

    jobs = search_remotive_jobs(keywords)
    if not jobs:
        return jsonify({"jobs": []}), 200

    # Gera embeddings das descrições de vagas
    descriptions = [job.get("description", "") for job in jobs]

    try:
        job_embeddings = get_embeddings(descriptions)
    except Exception as e:
        return jsonify({"error": f"Falha ao gerar embeddings das vagas: {e}"}), 500

    scored = []
    for job, emb in zip(jobs, job_embeddings):
        score = cosine_similarity(resume_embedding, emb)
        scored.append((score, job))

    # Ordena por similaridade e retorna até 5 mais relevantes
    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:5]

    result = []
    for score, job in top5:
        result.append({
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("candidate_required_location"),
            "description": clean_html(job.get("description")),
            "similarity": round(score, 4)
        })

    return jsonify({"status": "success", "jobs": result}), 200
