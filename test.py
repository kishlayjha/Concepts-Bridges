import subprocess
# subprocess.call(['java', '-jar', "/home/super-machine/NetBeansProjects/MeSHRanking/dist/MeSHRanking.jar"])
import numpy as np
import scipy.spatial
import sys

def cos_sim(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

embeddings = np.load('//home/super-machine/PycharmProjects/EmbeddingTransformation/files/embeddings/2016.npy')
# print(embeddings.shape)
# sys.exit(0)

embeddings_fishOils = np.asarray(embeddings[3541])
embeddings_raynaudDisease= np.asarray(embeddings[2598])

print(embeddings_fishOils.reshape(1,-1).shape)

bTermCosinePart1 = 1 - scipy.spatial.distance.cdist(embeddings_fishOils.reshape(1,-1), embeddings_raynaudDisease.reshape(1,-1), 'cosine')

# bTermCosinePart1 = cos_sim(embeddings_fishOils,embeddings_raynaudDisease)
print(bTermCosinePart1)