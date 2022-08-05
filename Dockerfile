FROM jupyter/pyspark-notebook
#Copy all files from the current directory to the /usr/local/spark directory
COPY /code /code
#Install requirements
#RUN pip install pandas