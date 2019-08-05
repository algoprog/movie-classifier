ARG PYTHON_VERSION="3.6"
FROM python:${PYTHON_VERSION}-stretch AS builder

RUN cd movie-clasifier && git fetch && pip install --upgrade .
RUN cp -r /movie-clasifier/model/ /usr/local/lib/python${PYTHON_VERSION}/site-packages/movie-clasifier/model

COPY . /movie-clasifier

WORKDIR /movie-clasifier

EXPOSE 80

CMD [ "python3", "./model.py", "--mode serve", "--port 80" ]
