import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(word_to_id)
print(id_to_word)

C = create_co_matrix(corpus, len(word_to_id))
print(C)

C0 = C[word_to_id['you']]
C1 = C[word_to_id['i']]

print(cos_similarity(C0, C1))

most_similar('you', word_to_id, id_to_word, C, top=5)

W = ppmi(C)
np.set_printoptions(precision = 3)
print('동시발생 행렬')
print(C)
print('-' * 50)
print('PPMI')
print(W)

U, S, V = np.linalg.svd(W)

print(C[0])
print(W[0])
print(U[0])
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
