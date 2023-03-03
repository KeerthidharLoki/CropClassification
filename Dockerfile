FROM tensorflow/tensorflow:2.1.1-gpu
COPY . /app
WORKDIR /app
RUN pip install -r requirments.txt
EXPOSE $PORT
CMD ["python","app.py"]
