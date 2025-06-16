from flask import Blueprint, request, jsonify
import requests
from semantic import get_embeddings, cosine_similarity
from resume_bp import resumes
from html.parser import HTMLParser
from itertools import combinations

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

def search_remotive_jobs(keywords):
    """
    Busca vagas do Remotive com base em combinações prioritárias de palavras-chave.
    Começa com todas as combinações de 3 termos, depois 2, até 1.
    """
    terms = set()
    for phrase in keywords:
        for word in phrase.lower().split():
            if len(word) > 3:
                terms.add(word)

    terms = list(terms)
    if not terms:
        return []

    # Tenta combinações do maior para o menor
    for size in range(min(5, len(terms)), 0, -1):
        combos = combinations(terms, size)
        for combo in combos:
            query = " ".join(combo)
            if query in job_cache:
                print(f"📦 Cache: {query}")
                return job_cache[query]

            print(f"🔍 Buscando combinação: {query}")
            try:
                resp = requests.get("https://remotive.com/api/remote-jobs", params={"search": query})
                resp.raise_for_status()
                jobs = resp.json().get("jobs", [])
                if jobs:
                    job_cache[query] = jobs
                    return jobs
            except Exception as e:
                print(f"❌ Erro ao buscar para '{query}': {e}")

    print("⚠️ Nenhuma combinação funcionou. Tentando termos individualmente...")

    jobs = []
    for term in terms:
        if term in job_cache:
            jobs += job_cache[term]
            continue

        print(f"🔹 Buscando individualmente: {term}")
        try:
            resp = requests.get("https://remotive.com/api/remote-jobs", params={"search": term})
            resp.raise_for_status()
            result = resp.json().get("jobs", [])
            job_cache[term] = result
            jobs += result
            if jobs:
                break
        except Exception as e:
            print(f"❌ Erro ao buscar '{term}': {e}")

    return jobs


    # Fallback final: termos individualmente
    print("⚠️ Nenhuma combinação deu resultado. Tentando termos individualmente...")
    jobs = []
    for term in terms:
        if term in job_cache:
            jobs += job_cache[term]
            continue

        print(f"🔹 Buscando individualmente: {term}")
        try:
            resp = requests.get("https://remotive.com/api/remote-jobs", params={"search": term})
            resp.raise_for_status()
            result = resp.json().get("jobs", [])
            job_cache[term] = result
            jobs += result
            if jobs:
                break
        except Exception as e:
            print(f"❌ Erro ao buscar '{term}': {e}")

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

    # Ordena e retorna as vagas mais similares
    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = scored[:3]

    result = []
    for score, job in top3:
        result.append({
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("candidate_required_location"),
            "description": clean_html(job.get("description")),
            "similarity": round(score, 4)
        })

    return jsonify({"status": "success", "jobs": result}), 200
