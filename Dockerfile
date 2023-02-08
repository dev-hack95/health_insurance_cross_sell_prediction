FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN dvc repro
EXPOSE 8501
CMD streamlit run app.py