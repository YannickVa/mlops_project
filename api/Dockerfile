# Stage 1: Builder
# This stage installs dependencies using Poetry and creates a requirements.txt file.
FROM python:3.9 AS builder

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install poetry==1.5.1

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Final Production Image
# This stage creates the lean final image.
FROM python:3.9-slim

WORKDIR /app

ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG UID=1001
ARG GID=1001

RUN groupadd -g ${GID} ${APP_GROUP} && \
    useradd -u ${UID} -g ${APP_GROUP} -m -s /bin/bash ${APP_USER}

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api /app/api
COPY ./ml/artifacts /app/ml/artifacts

RUN chown -R ${APP_USER}:${APP_GROUP} /app
USER ${APP_USER}

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]