FROM apache/airflow:latest

ARG REPO=NatuReddit

USER root

# Get external data

RUN rm -rf credentials \
    && rm -rf userscripts

COPY credentials/ credentials/
COPY userscripts/ userscripts/

RUN apt-get update \
    && apt-get install -y git
    # && ./userscripts/gh_install.sh \


USER airflow

# Clone repository
RUN git config --global user.email "who" \
    && git config --global user.name "cares" \
    && export GH_TOKEN=$(cat ./credentials/gh_token.txt) \
    && git clone https://$GH_TOKEN@github.com/DiegoBiagini/${REPO}.git

# Setup python environment

RUN cd ${REPO}/ \
    && pip install -r requirements.txt \
    && dvc pull