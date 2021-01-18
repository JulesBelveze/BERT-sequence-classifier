# use python
FROM python:3.7-slim

# ARGS
ARG checkpoint=checkpoints/roberta_80000.bin
ARG quantize=true

# copy needed files
COPY utils/ /utils/
COPY models/ models/
COPY ${checkpoint} ${checkpoint}
COPY ["inferer.py", "requirements_inference.txt", "app.py", ".env", "./"]

# installed needed packages
RUN pip3 install --no-cache-dir -r requirements_inference.txt

# set current dir as working dir
WORKDIR /

ENV PORT 8080

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers=1 --timeout=200 --worker-tmp-dir /dev/shm app:app