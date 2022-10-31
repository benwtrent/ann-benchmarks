"""
ann-benchmarks interface for Apache Lucene.
"""

import sklearn.preprocessing
import numpy as np
from multiprocessing.pool import ThreadPool


import lucene
from lucene import JArray
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import KnnVectorQuery, IndexSearcher
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, VectorSimilarityFunction, DirectoryReader
from org.apache.lucene.codecs.lucene91 import Lucene91Codec, Lucene91HnswVectorsFormat
from org.apache.lucene.document import Document, FieldType, KnnVectorField, StoredField

from ann_benchmarks.algorithms.base import BaseANN


class Codec(Lucene91Codec):
    def __init__(self, M, efConstruction):
        super(Codec, self).__init__()
        self.M = M
        self.efConstruction = efConstruction

    def getKnnVectorsFormatForField(self):
        Lucene91HnswVectorsFormat(self.M, self.efConstruction)


class LucenePyLucene(BaseANN):
    """
    KNN using the Lucene Vector datatype.
    """

    def __init__(self, metric: str, dimension: int, param):
        lucene.initVM(vmargs=['-Djava.awt.headless=true -Xmx6g -Xms6g'])
        self.metric = metric
        self.dimension = dimension
        self.param = param
        self.short_name = f"lucenepyknn-{dimension}-{param['M']}-{param['efConstruction']}"
        self.n_iters = -1
        self.train_size = -1
        self.simFunc = VectorSimilarityFunction.DOT_PRODUCT if self.metric == "angular" \
            else VectorSimilarityFunction.EUCLIDEAN
        if self.metric not in ("euclidean", "angular"):
            raise NotImplementedError(f"Not implemented for metric {self.metric}")

    def done(self):
        if self.dir:
            self.dir.close()

    def fit(self, X):
        # X is a numpy array
        if self.dimension != X.shape[1]:
            raise Exception(f"Configured dimension {self.dimension} but data has shape {X.shape}")
        if self.metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self.train_size = X.shape[0]
        iwc = IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        codec = Codec(self.param['M'], self.param['efConstruction'])
        iwc.setCodec(codec)
        iwc.setRAMBufferSizeMB(1994.0)
        self.dir = FSDirectory.open(Paths.get(self.short_name + ".index"))
        iw = IndexWriter(self.dir, iwc)
        fieldType = KnnVectorField.createFieldType(self.dimension, self.simFunc)
        id = 0
        X = X.tolist()
        for x in X:
            doc = Document()
            doc.add(KnnVectorField("knn", JArray('float')(x), fieldType))
            doc.add(StoredField("id", id))
            id += 1
            iw.addDocument(doc)
            if id + 1 % 1000 == 0:
                print(f"written {id} docs")
        iw.forceMerge(1)
        print(f"written {id} docs")
        iw.close()
        self.searcher = IndexSearcher(DirectoryReader.open(self.dir))

    def set_query_arguments(self, fanout):
        self.name = f"lucenepyknn dim={self.dimension} {self.param}"
        self.fanout = fanout

    def query(self, q, n):
        if self.metric == 'angular':
            q = q / np.linalg.norm(q)
        return self.run_knn_query(n + self.fanout, n, q.tolist())

    def prepare_batch_query(self, X, n):
        if self.metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        self.queries = X.tolist()
        self.topK = n

    def run_batch_query(self):
        self.res = []
        append = self.res.append
        n = self.topK
        num_candidates = self.topK + self.fanout
        for q in self.queries:
            append(self.run_knn_query(num_candidates=num_candidates, n=n, q=q))

    def get_batch_results(self):
        return self.res

    def run_knn_query(self, num_candidates, n, q):
        query = KnnVectorQuery("knn", JArray('float')(q), num_candidates)
        topdocs = self.searcher.search(query, n)
        return [d.doc for d in topdocs.scoreDocs]

    def batch_query(self, X, n):
        """Provide all queries at once and let algorithm figure out
           how to handle it. Default implementation uses a ThreadPool
           to parallelize query processing."""
        pool = ThreadPool()
        if self.metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        X = X.tolist()
        num_candidates = n + self.fanout
        self.res = pool.map(lambda q: self.run_knn_query(num_candidates=num_candidates, n=n, q=q), X)
