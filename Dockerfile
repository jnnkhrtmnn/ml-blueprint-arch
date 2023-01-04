FROM python:3.10.9-slim

RUN pip install pip==22.3.1
RUN pip install poetry==1.2.2

RUN mkdir /code
WORKDIR /code


COPY ./poetry.lock /code/poetry.lock
COPY ./pyproject.toml /code/pyproject.toml

COPY ./ml_project /code/ml_project

RUN poetry install --without dev

COPY ./models /code/models

CMD ["poetry", "run", "uvicorn", "ml_project.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
