import sys
import json
import re
import copy
import math
import csv
import pandas as pd
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


# defining the Jaccard distance
def jaccard(a, b):
    inter = list(set(a) & set(b))
    I = len(inter)
    union = list(set(a) | set(b))
    U = len(union)
    return round(1 - (float(I) / U), 4)

def distance(a,b):
    dist = 0
    for i in range(0,len(a)):
        dist += math.pow(a[i]-b[i],2)

    return math.sqrt(dist)

# k-means implementation
def kmeans(id, centroids, terms_all, l, k=10):
    count = 0

    for h in range(0,50):
        indices = []
        cen_txt = []
        count = count + 1
        # lay ra index cua tung tam cum
        for item in centroids:
            indices.append(id.index(item))
        # lay ra term cua tung tam cum
        for x in indices:
            cen_txt.append(terms_all[x])

        # mang chua nhan cua cum (0-24) tuong ung cho tung diem du lieu trong id
        cluster = []
        for i in range(l):
            d = []
            for j in range(k):
                d.append(jaccard(terms_all[i], cen_txt[j]))

            ans = d.index(min(d))
            cluster.append(ans)

        centroids1 = up_date(id, cluster, terms_all, l, k)

        centroids = copy.deepcopy(centroids1)

    return cluster,centroids




# updating the centroids at every iteration
def up_date(id, cluster, terms_all, l, k):
    indices = []
    new_centxt_index = []
    new_centroid = []

    for i in range(k):
        # lay ra cac diem cung 1 cum
        # j la key, u la value
        x = []
        for j, u in enumerate(cluster):
            if (u == i):
                x.append(j)
        indices.append(x)

        # lay ra 1 mang thuoc cum i
        m = indices[i]

        if (len(m) != 0):
            txt = []
            # lay ra term cua tung phan tu trong tung cum
            for p in m:
                txt.append(terms_all[p])

            # tinh khoang cac cua tung phan tu den cac phan tu con lai
            sim = []
            for i in range(len(m)):
                a = []
                for j in range(len(m)):
                    a.append(jaccard(txt[i].tolist(), txt[j].tolist()))
                sim.append(a)
            # tinh tong khoang cach tu 1 phan tu den cac phan tu con lai trong cum

            f1 = []
            for i in sim:
                f1.append(sum(i))

        # lay ra index cua gia tri nho nhat trong f1
        minSumID = f1.index(min(f1))
        new_centxt_index.append(m[minSumID])

    for x in new_centxt_index:
        new_centroid.append(id[x])

    return new_centroid

def loadInit():
    # doc file khoi tao tam
    text_file = open('./init.txt', "r")
    centroids = text_file.read().split(',')
    centroids = [x.strip('\n') for x in centroids]
    centroids = [int(x) for x in centroids]  # ids of centroids

    return centroids
def loadData():
    centroids = loadInit()
    #data.csv
    df = pd.read_csv("./twitter.csv", encoding='ISO-8859-1', sep=',')
    id = df['ID'].values.tolist()
    terms_all = df['comment'].values.tolist()

    cen = []
    # index trong id chua tam cum
    for i in centroids:
        cen.append(id.index(i))

    vectorizer = HashingVectorizer(n_features=2 ** 4)
    X = vectorizer.fit_transform(terms_all)
    terms = X.toarray()
    return terms, id, centroids



if __name__ == '__main__':

    start = time.time()
    terms_all,id,cen = loadData()

    N = len(id)
    K = len(cen)

    cluster,centroids = kmeans(id, cen, terms_all, N,K)
    end = time.time()
    print('total run-time: %f ms' % ((end - start) * 1000))


