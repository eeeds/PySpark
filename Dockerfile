FROM apache/spark-py
#Copy all files from the current directory to the /usr/local/spark directory
COPY . /usr/local/spark