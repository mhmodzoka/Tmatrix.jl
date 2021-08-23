
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

"""
Create a single index from m and n
We fill the index like this:
`
idx = 0
for n = 1:n_max    
    for m = -n:n
        global idx
        idx += 1
    end
end
`
"""
def single_index_from_m_n(m: int, n: int):
    return n * (n + 1) + m
end

"""
    Get the maximum single index, given the maximum n.
"""
def get_max_single_index_from_n_max(n_max: int):
    return single_index_from_m_n(n_max, n_max)