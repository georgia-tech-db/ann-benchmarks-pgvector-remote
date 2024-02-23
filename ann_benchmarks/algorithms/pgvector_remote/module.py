import subprocess
import sys
import numpy as np

from numpy.core.multiarray import array as array
import pgvector.psycopg
import psycopg
# from pyscopg.errors import _info_to_dict
from pprint import pprint

from ..base.module import BaseANN


class PGVector_Remote(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
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
        pprint(notice.__reduce__())

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="postgres", password="postgres", dbname="ann", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        # send client messages to stdout
        conn.add_notice_handler(self.notice_handler)


        cur = conn.cursor()
        # cur.execute("SET client_min_messages = debug1")

        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        # print a couple random rows
        cur.execute("SELECT * FROM items ORDER BY random() LIMIT 5")
        print("sample rows:")
        for row in cur.fetchall():
            print(row)
        print("creating index...")
        if self._metric == "angular":
            # cur.execute(
                # "CREATE INDEX ON items USING pinecone (embedding vector_cosine_ops) WITH (spec = '{\"serverless\":{\"cloud\":\"aws\",\"region\":\"us-west-2\"}})" % (self._m, self._ef_construction)
            # )
            pass
        elif self._metric == "euclidean":
            print('hello Euclid')
            cur.execute("SET ivfflat.probes = 17")
            cur.execute("SET pinecone.api_key = '5b2c1031-ba58-4acc-a634-9f943d68822c'")
            cur.execute("SHOW ivfflat.probes")
            print(cur.fetchone())
            cur.execute("SHOW pinecone.api_key")
            print(cur.fetchone())
            # set client debug level to debug1
            cur.execute("CREATE INDEX pc_index ON items USING pinecone (embedding) WITH (spec = '{\"serverless\":{\"cloud\":\"aws\",\"region\":\"us-west-2\"}}')")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
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
        import json
        async def async_query(pool, vec, topK):
            async with pool.acquire() as conn:
                print("vec: ", vec)
                vec_str = json.dumps(vec.tolist())
                print("vec_str: ", vec_str)
                print(await conn.fetch("SELECT * FROM test ORDER BY vec <-> %s LIMIT %s", vec_str, topK))

        async def run_async_queries(vecs, topK):
            pool = await asyncpg.create_pool(user='postgres', password='postgres', database='ann', min_size=2, max_size=20)
            await asyncio.gather(*[async_query(pool, vec, topK) for vec in vecs])
            await pool.close()

        # run the async queries
        return asyncio.run(run_async_queries(vecs=X, topK=n))


    def get_memory_usage(self):
        if self._cur is None:
            return 0
        # self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        # return self._cur.fetchone()[0] / 1024
        return 1

    def __str__(self):
        return f"PGVector_Remote(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
