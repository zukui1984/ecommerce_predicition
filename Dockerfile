FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY predict.py model.json dv.pkl ./

EXPOSE 9696

ENTRYPOINT ["python", "predict.py"]