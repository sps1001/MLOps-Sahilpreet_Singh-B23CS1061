FROM python:3.10
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "ml_dl_ops_ass_3_fine_tuning_classification.py"]