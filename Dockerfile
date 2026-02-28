FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-image/matplotlib rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /app/
COPY semquant /app/semquant
COPY app.py /app/app.py

RUN pip install -U pip && pip install .[ui]

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
