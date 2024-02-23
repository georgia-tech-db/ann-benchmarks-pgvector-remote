FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all libcurl4-openssl-dev

# set remote origin
RUN git clone -b ann_benchmarks_dev https://github.com/georgia-tech-db/pgvector-remote.git /tmp/pgvector-remote
RUN pip install psycopg[binary] pgvector

RUN echo a012345678
# fetch (cd into the directory and then fetch)
RUN cd /tmp/pgvector-remote && \
    git pull origin

RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pgvector-remote && \
	make clean && \
	make -j 11 OPTFLAGS="-march=native -mprefer-vector-width=512" && \
	make install


USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann' SUPERUSER" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '4GB'"
USER root
