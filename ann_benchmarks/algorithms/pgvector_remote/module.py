import subprocess
import sys
import numpy as np
import time

from numpy.core.multiarray import array as array
import pgvector.psycopg
import pgvector.asyncpg
import psycopg
# from pyscopg.errors import _info_to_dict
from pprint import pprint

from ..base.module import BaseANN

# get pinecone api key from caller's dotenv .env file
import os
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")




class PGVector_Remote(BaseANN):
    def __init__(self, metric, arg_group):
        self._metric = metric
        # self._m = arg_group['M']
        # self._ef_construction = arg_group['efConstruction']
        self._cur = None
        self.global_count = 0

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def notice_handler(self, notice):
        print("Received notice:", notice.message_primary)
        # user pprint to print the notice as a dictionary
        # pprint(notice.__reduce__())

    def fit(self, X):
        # subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr) # TODO: we'd rather do this in the Dockerfile
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        # send client messages to stdout
        conn.add_notice_handler(self.notice_handler)


        cur = conn.cursor()
        # cur.execute("SET client_min_messages = debug1")

        # drop
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        if self._metric == "angular":
            # cur.execute(
                # "CREATE INDEX ON items USING pinecone (embedding vector_cosine_ops) WITH (spec = '{\"serverless\":{\"cloud\":\"aws\",\"region\":\"us-west-2\"}})" % (self._m, self._ef_construction)
            # )
            pass
        elif self._metric == "euclidean":
            print('hello Euclid')
            cur.execute("ALTER SYSTEM SET pinecone.api_key = '%s'" % PINECONE_API_KEY)
            cur.execute("SHOW pinecone.api_key")
            print(cur.fetchone())
            # set client debug level to debug1
            cur.execute("SET client_min_messages = debug1")
            cur.execute("CREATE INDEX pcindex ON items USING pinecone (embedding) WITH (spec = '{\"serverless\":{\"cloud\":\"aws\",\"region\":\"us-west-2\"}}')")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        # sleep 15s to allow the index to build
        print("sleeping for 15s")
        time.sleep(15)
        print("done!")
        self._cur = cur

    def set_query_arguments(self, *args):
        pass
        # self._ef_search = ef_search
        # self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self.global_count += 1
        print(f"global_count: {self.global_count}")
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        neighbors =  [id for id, in self._cur.fetchall()]
        print(f"neighbors: {neighbors}")
        return neighbors

    def batch_query(self, X: np.array, n: int) -> None:
        print("batch_query")
        import asyncpg
        import asyncio

        async def init(conn):
            await pgvector.asyncpg.register_vector(conn)

        async def async_query(pool, vec, topK):
            async with pool.acquire() as conn:
                start = time.time()
                neighbors = await conn.fetch("SELECT id,embedding<-> $1 FROM items ORDER BY embedding <-> $1 LIMIT $2", vec, topK)
                latency = time.time() - start
                # print(latency)
                return {'neighbor_list': [n['id'] for n in neighbors], 'latency': latency}

        async def run_async_queries(vecs, topK):
            pool = await asyncpg.create_pool(user='ann', password='ann', database='ann', min_size=2, max_size=60, init=init)
            results = await asyncio.gather(*[async_query(pool, vec, topK) for vec in vecs])
            await pool.close()
            return results

        # run the async queries
        result = asyncio.run(run_async_queries(vecs=X, topK=n))
        self.res = [r['neighbor_list'] for r in result]
        self.batch_latencies = [r['latency'] for r in result]
        return result

    def get_batch_latencies(self):
        return self.batch_latencies


    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('pcindex')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGVector_Remote()"
