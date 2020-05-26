import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

import sys
import json
import re
import copy
import math
import csv
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

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


mod = SourceModule("""

__device__ float distE(int i,int j,int size, float * terms){
    float dist;
    for(int t =0;t<size;t++){
        dist += pow(terms[i *size +t]-terms[j*size+t],2);
    }
    
    return sqrt(dist);
}

__global__ void kmeans(int * id, float * terms, int * centroids,int * new_centroids, float * centroids_terms, int * cluster, float * dist){
    unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;
    unsigned int size = blockDim.x;
    
    int t = centroids[j];
    //dist[i * blockDim.x + j] = sqrt(pow(terms[i] - centroids_terms[j],2 ));
    //dist[i * blockDim.x + j] = sqrt(pow(terms[i] - terms[centroids[j]],2 ));
    dist[i * size + j] = distE(i,t,size,terms);
    __syncthreads();
    
    if (j == 0){
        float min_centroid = dist[i * blockDim.x + j];
        float min_index = 0;
        
        for (int m = 0; m < blockDim.x; m++){
            if (dist[i * blockDim.x + m] < min_centroid){
                min_centroid = dist[i * blockDim.x + m];
                min_index = m;
                
            }
        }
        cluster[i] = min_index;
        
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
""")

kmean_cuda = mod.get_function("kmeans")

recompute_clusters = mod.get_function('recompute_clusters')
reset = mod.get_function('reset_values')

def kmeans_iter(id, terms, centroids,new_centroids, centroids_terms,cluster,dist):

    kmean_cuda(id,terms,centroids,new_centroids,centroids_terms,cluster,dist, block=(K, 1, 1), grid=(N, 1))

    recompute_clusters(id,terms,centroids,new_centroids,centroids_terms,cluster,dist, block=(N, 1, 1), grid=(K, 1))
    reset(new_centroids, block=(N, 1, 1), grid=(K, 1))

def kmean_pycuda(id, terms, centroids, N, K, model):
    terms_cpu = []
    text = []
    for i in range(0, len(terms)):
        t = " ".join(terms[i])
        text.append(t)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)

    for i in range(0, len(terms)):
        vector = vectorizer.transform([text[i]])
        terms_cpu.append(vector.toarray().tolist()[0])
        #terms_cpu.append(np.random.random(5).astype(np.float32).tolist())



    centroids_cpu = []

    for i in cen:
        vector = vectorizer.transform([text[i]])
        centroids_cpu.append(vector.toarray().tolist()[0])



    id_gpu = gpuarray.to_gpu(np.asarray(id).astype(np.int32))
    terms_gpu = gpuarray.to_gpu(np.asarray(terms_cpu).astype(np.float32))
    centroids_terms = gpuarray.to_gpu(np.asarray(centroids_cpu).astype(np.float32))
    centroids = gpuarray.to_gpu(np.asarray(centroids).astype(np.int32))
    new_centroids = gpuarray.to_gpu(np.zeros((K,N)).astype(np.int32))
    cluster = gpuarray.to_gpu(np.empty(N).astype(np.int32))
    dist = gpuarray.to_gpu(np.zeros((N, K)).astype(np.float32))

    for i in range(0,500):
        kmeans_iter(id_gpu,terms_gpu,centroids,new_centroids,centroids_terms,cluster,dist)

    return centroids,dist,cluster

def convertWordToVec(model, word):
    return model[word]





def loadGloveModel(gloveFile):
    print("Loading Glove Model")

    with open(gloveFile, encoding="utf8") as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


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


if __name__ == "__main__":


    file = "./glove.6B/glove.6B.50d.txt"
    # model = loadGloveModel(file)
    model = 1
    # print(type(model['hello world']))

    id, terms = loadData()
    centroids = loadInit()


    cen = []
    #index trong id chua tam cum
    for i in centroids:
        cen.append(id.index(i))


    N = len(id)
    K= len(centroids)
    print('before')
    print(cen)


    centroids,dist,cluster = kmean_pycuda(id, terms, cen, N, K, model)
    print('after')
    print(centroids)
    print(cluster)





