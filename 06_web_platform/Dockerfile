# AnaeroBio-Hub v4.0 — production image (build context = repo root; see .dockerignore allowlist)
#   docker compose -f 06_web_platform/docker-compose.yml up --build -d
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libxrender1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first (cached layer). Strict: any failure here fails the build.
COPY 06_web_platform/requirements.txt /tmp/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt
# 2) RDKit is optional (model_engine.py falls back to built-in salt stripping) → never fail the build on it.
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "rdkit>=2024.9.1" || echo "[WARN] rdkit not installed — using built-in desalting fallback"

# 3) Application + the 3 core production files (allowed through .dockerignore).
#    docker-compose.yml bind-mounts the host copies over these at runtime so DB/bundle updates persist.
COPY 06_web_platform/           ./06_web_platform/
COPY 01_final_dataset/          ./01_final_dataset/
COPY 03_priors_and_metabolism/  ./03_priors_and_metabolism/

# Fail the BUILD (not the first request) if a core file is missing; normalise CRLF from Windows checkouts.
RUN test -s 01_final_dataset/anaerobic_multimodal_benchmark.db \
 && test -s 03_priors_and_metabolism/QSAR_Observed_Microbial_Metabolism_Pathways.csv \
 && test -s 06_web_platform/multimodal_v4_bundle.pkl \
 && test -s 06_web_platform/ui/common.py \
 && test -s 06_web_platform/.streamlit/config.toml \
 && sed -i 's/\r$//' 06_web_platform/docker_entrypoint.sh \
 && python -c "import pickle,sklearn,numpy; b=pickle.load(open('06_web_platform/multimodal_v4_bundle.pkl','rb')); print('bundle OK:', b['version'], '| sklearn', sklearn.__version__, '| numpy', numpy.__version__)"

ENV PYTHONPATH=/app/06_web_platform
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/status', timeout=8).status==200 else 1)"

CMD ["bash", "/app/06_web_platform/docker_entrypoint.sh"]
