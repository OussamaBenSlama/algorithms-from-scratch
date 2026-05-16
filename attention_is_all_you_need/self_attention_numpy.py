import numpy as np


def softmax(M , axis=0):
    e_M = np.exp(M-np.max(M, axis=axis, keepdims=True)) #numerical stability (prevents overflow)
    return e_M / np.sum(e_M,axis=axis, keepdims=True)

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[1]

    soft_score = softmax((np.dot(Q,np.array(K).transpose()))/d_k**0.5, axis=1)

    return np.dot(soft_score,V)

