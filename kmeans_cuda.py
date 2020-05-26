import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pandas as pd

import sys
import json
import re
import copy
import math
import csv
from sklearn.feature_extraction.text import HashingVectorizer
import time

# intial data cleaning and preprocessing using regular expressions

regex_str = [
    r'<[^>]+>',
    r'(?:@[\w_]+)',
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',
    r"(?:[a-z][a-z'\-_]+[a-z])",
    r'(?:[\w_]+)',
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=True):
    # tach chuoi ban dau dua vao khoang trang, chuyen ve chu thuong=>tra ve mang
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


mo = """

__device__ float distE(int i,int j,int size, float * terms){
    float dist;
    for(int t =0;t<size;t++){
        dist += pow(terms[i *size +t]-terms[j*size+t],2);
    }
    return sqrt(dist);
}


__global__ void kmeans(int * id, float * terms, int * centroids,int * new_centroids, float * centroids_terms, int * cluster, float dist[%(N)s][%(K)s]){
    unsigned int Bx = blockIdx.x;
    unsigned int By = blockIdx.y;
    unsigned int Tx = threadIdx.x;
    unsigned int Ty = threadIdx.y;
    unsigned int BSx = blockDim.x;
    unsigned int BSy = blockDim.y;
    unsigned int GSx = gridDim.x;
    unsigned int GSy = gridDim.y;
    int i = Tx + Bx*BSx ;
    int j = Ty + By*BSy ;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;  

    dist[c][r] = distE(r,centroids[0],16,terms);
    // dist[ r * %(K)s + c] = 1;
    __syncthreads();
    if(Bx ==0 && Tx==0){
        float min_centroid = dist[c][0];
        float min_index = 0;

        for (int m = 0; m < %(K)s; m++){
            if (dist[c][m] < min_centroid){
                min_centroid = dist[c][m];
                min_index = m;

            }
        }
        cluster[r*%(K)s + c] = min_index;
    }


}

__global__ void recompute_clusters(int * id, float * terms, int * centroids,int * new_centroids, float * centroids_terms, int * cluster, float * dist){
    unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;

    if(cluster[j] == i){
        new_centroids[i*blockDim.x +j] = 1;

    }
    __syncthreads();

    if(j ==0){
        for(int t =0;t<blockDim.x;t++){
            if(new_centroids[i*blockDim.x +t] ==1){
                centroids[i] = t; 
                break;
            }
        }

    }

}
__global__ void reset_values(int *new_centroids){
    unsigned int i = threadIdx.x;
    unsigned int j = blockIdx.x;
    unsigned int size = blockDim.x;

    new_centroids[j*size +i] = 0;
}
"""


def kmeans_iter(id, terms, centroids, new_centroids, centroids_terms, cluster, dist):
    kmean_cuda(id, terms, centroids, new_centroids, centroids_terms, cluster, dist, block=(32, 32, 1), grid=(10, 1024))
    recompute_clusters(id, terms, centroids, new_centroids, centroids_terms, cluster, dist, block=(32, 32, 1),
                       grid=(10, 1024))
    reset(new_centroids, block=(32, 32, 1), grid=(10, 1024))


def kmean_pycuda(id, terms, centroids, N, K):
    terms_cpu = []
    text = []
    # for i in range(0, len(terms)):
    #     t = " ".join(terms[i])
    #     text.append(t)
    #
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(text)

    # for i in range(0, len(terms)):
    #     vector = vectorizer.transform([text[i]])
    #     #terms_cpu.append(vector.toarray().tolist()[0])
    #     terms_cpu.append(np.random.random(5).astype(np.float32).tolist())

    centroids_cpu = []

    for i in centroids:
        centroids_cpu.append(np.random.random(5).astype(np.float32).tolist())

    # id comment
    id_gpu = gpuarray.to_gpu(np.asarray(id).astype(np.int32))
    # comment convert -> vector
    terms_gpu = gpuarray.to_gpu(terms.astype(np.float32))
    # comment center convert -> vector
    centroids_terms = gpuarray.to_gpu(np.asarray(centroids_cpu).astype(np.float32))
    # index center in list id comment
    centroids = gpuarray.to_gpu(np.asarray(centroids).astype(np.int32))
    # new_centroids
    new_centroids = gpuarray.to_gpu(np.zeros((K, N)).astype(np.int32))
    # cluster
    cluster = gpuarray.to_gpu(np.empty(N).astype(np.int32))
    # dist
    dist = gpuarray.to_gpu(np.zeros((N, K)).astype(np.float32))

    for i in range(0, 1):
        kmeans_iter(id_gpu, terms_gpu, centroids, new_centroids, centroids_terms, cluster, dist)
    #
    return centroids, dist, cluster


def loadData():
    file = open('./twitter.csv', 'r')
    data = csv.reader(file)
    id = []
    terms_all = []
    for row in data:
        content = row[5]
        index = int(row[1])
        tokens = preprocess(content)
        terms_all.append(tokens)
        id.append(index)

    return (id, terms_all)


def loadInit():
    # doc file khoi tao tam
    text_file = open('./init.txt', "r")
    centroids = text_file.read().split(',')
    centroids = [x.strip('\n') for x in centroids]
    centroids = [int(x) for x in centroids]  # ids of centroids

    return centroids


def load():
    centroids = loadInit()
    df = pd.read_csv("./data.csv", encoding='ISO-8859-1', sep=',')
    id = df['ID'].values.tolist()
    terms_all = df['comment'].values.tolist()

    cen = []
    # index trong id chua tam cum
    for i in centroids:
        cen.append(id.index(i))

    vectorizer = HashingVectorizer(n_features=2 ** 4)
    X = vectorizer.fit_transform(terms_all)
    terms = X.toarray()

    return terms, id, cen


def printResult(clusters, centroids, id, K):
    fileCluster = open('cluster.txt', 'w+')
    fileCentroid = open('centroid.txt', 'w+')

    # centroids
    list_IDcentroids = centroids.get().tolist()
    list_centroids = map(lambda x: id[x], list_IDcentroids)
    values = ','.join(map(str, list_centroids))
    fileCentroid.write(values)

    # clusters
    list_clusters = clusters.get().tolist()
    result_cluster = []

    final = []
    for i in range(K):
        x = []
        for j, u in enumerate(list_clusters):
            if u == i:
                x.append(id[j])
        final.append(x)
        result = ",".join(map(str, x))
        fileCluster.writelines(str(K) + "=>[" + result + "]")


start = time.time()
terms, id, cen = load()

N = len(id)
K = len(cen)

kernel_code = mo % {
    'N': N,
    'K': K
}
mod = SourceModule(kernel_code)

kmean_cuda = mod.get_function("kmeans")
recompute_clusters = mod.get_function('recompute_clusters')
reset = mod.get_function('reset_values')
centroids, dist, cluster = kmean_pycuda(id, terms, cen, N, K)

end = time.time()
printResult(cluster, centroids, id, K)
print( 'total run-time: %f ms' % ((end - start) * 1000))


