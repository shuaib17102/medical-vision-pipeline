FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY model.py app.py ./
COPY models/best_model.pth models/best_model.pth

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]