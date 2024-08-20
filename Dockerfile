#base image
FROM python:3.12

#working dir
WORKDIR /app

#copy
COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

#run
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

#expose
EXPOSE 5000

#command
# CMD [ "python", "app.py" ]
CMD [ "gunicorn", "-b", "0:0:0:0:5000", "app:app" ]
