FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "label_platform1.py", "--server.port=8501", "--server.address=0.0.0.0"]