import numpy as np

def hamming_dist(bitvec1, bitvec2):
    '''
    int_bitvec1 = 0
    for i in bitvec1:
        int_bitvec1 |= int(i) & 1
        int_bitvec1 <<= 1

    int_bitvec2 = 0
    for i in bitvec2:
        int_bitvec2 |= int(i) & 1
        int_bitvec2 <<= 1

    xor = int_bitvec1 ^ int_bitvec2
    res = 0

    while(xor > 0):
        xor &= xor - 1
        res += 1

    return res
    '''
    return np.count_nonzero(bitvec1!=bitvec2)

def compute_hamming_dist(query, database):
    
    #res = [[[] for _ in range (len(database))] for _ in range (len(query))]
    res = np.empty(shape=(len(query), len(database)), dtype=int)
    #for i in range (len(query)):
        #for j in range (len(database)):
            #res[i][j] = hamming_dist(query[i],database[j])
    it = np.nditer(res, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = hamming_dist(query[it.multi_index[0]], database[it.multi_index[1]])
        it.iternext()
    

    return res

def top_k(k, query, database):

    res = compute_hamming_dist(query, database)
    top = np.argsort(res, axis = 1)

    top = np.array([i[:k:1] for i in top])
    return top

def reli(k, query_emb, query_lab, database_emb, database_lab):
    top = top_k(k, query_emb, database_emb)

    database_lab = np.argmax(database_lab,1)
    topk_lab = np.take(database_lab, top)
    query_lab = np.argmax(query_lab,1)
    
    bool_vals = np.equal(np.expand_dims(query_lab, 1), topk_lab)

    precision = np.mean(bool_vals, 1)

    return np.mean(precision)

if __name__ == "__main__":
    query_emb = np.array(np.random.randint(2,size = (4,5)), dtype = np.float)
    database_emb = np.array(np.random.randint(2,size = (14,5)), dtype = np.float)

    query_lab = np.random.randint(0, 3, size = 4)
    one_hot = np.zeros((4,3))
    one_hot[np.arange(4), query_lab] = 1
    query_lab = one_hot

    database_lab = np.random.randint(0, 3, size = 14)
    one_hot = np.zeros((14,3))
    one_hot[np.arange(14), database_lab] = 1
    database_lab = one_hot

    print(reli(2, query_emb, query_lab, database_emb, database_lab))




