FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY core/ ./core/
RUN pip install -e "."
COPY . .
ENTRYPOINT ["python", "apps/cli/rank.py"]