FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/home/user/app/.cache/models
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user recommender ./recommender
COPY --chown=user data ./data
COPY --chown=user web ./web
COPY --chown=user eval ./eval

# Download the embedding model at build time so the first request is fast.
RUN python -c "from recommender.config import get_settings; from recommender.service import Recommender; r = Recommender(get_settings()); assert r.embedder.name != 'hashing', 'embedding model failed to load'"

EXPOSE 7860
# Render (and most hosts) set PORT; default to 7860 for local runs.
CMD ["sh", "-c", "exec uvicorn recommender.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
