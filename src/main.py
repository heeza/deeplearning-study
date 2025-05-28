import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar

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