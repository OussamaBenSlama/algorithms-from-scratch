import torch


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    omega = torch.matmul(Q,K.T)
    d_k = Q.shape[1]

    soft_score = torch.softmax(
        omega / (d_k ** 0.5),
        dim=1
    )

    return torch.matmul(soft_score, V)
