FROM continuumio/miniconda3

WORKDIR /app

RUN conda install -y -c conda-forge \
    python=3.10 \
    rdkit \
    streamlit \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
"--server.port=8501", \
"--server.address=0.0.0.0", \
"--server.headless=true"]