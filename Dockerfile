FROM python:3.8-slim

COPY requirements.txt /root/requirements.txt

RUN chown -R root:root /root

WORKDIR /root
RUN pip3 install -r requirements.txt

COPY ./templates/ ./templates/
COPY ensembles.py .
COPY ml_server.py .
COPY run.py .
COPY ./server/ ./server/
COPY ./static/ ./static/

ENV SECRET_KEY prod
ENV FLASK_APP run.py

RUN chown -R root:root /root
RUN chmod +x ./run.py
CMD ["python3", "./run.py"]
