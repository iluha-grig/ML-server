FROM python:3.8-slim

COPY requirements.txt /root/requirements.txt

RUN chown -R root:root /root

WORKDIR /root
RUN pip3 install -r requirements.txt

COPY server/ ./server/

ENV SECRET_KEY prod
ENV FLASK_APP run.py

RUN chown -R root:root /root
RUN chmod +x server/flaskr/run.py
CMD ["python3", "server/flaskr/run.py"]
