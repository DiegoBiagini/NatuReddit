FROM pytorch/torchserve:0.6.0-cpu

USER root

EXPOSE 8080
EXPOSE 8081
COPY torchserve/ torchserve
COPY serve_requirements.txt requirements.txt
# Just for debugging purposes
RUN apt-get update
RUN apt-get install -y curl

RUN pip install -r requirements.txt
CMD ["torchserve", "--start", "--model-store", "torchserve/models", "--ts-config" ,"torchserve/config.properties", "--models", "naturedditTI=naturedditTI.mar"]