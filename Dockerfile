FROM python:3.8.5 as builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip & pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.8.5 as final
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

ADD ./app /app
WORKDIR /app 
EXPOSE 8180
EXPOSE 8181
VOLUME /app/models

ENTRYPOINT ["python", "/app/run_server.py"]

# docker run -d -p 8180:8180 -p 8181:8181 -v  E:\PROJECTS\Directum_test_task\models:/models doc_analizer 
# docker build --no-cache -t doc_analizer .
# docker images
# docker ps -a
