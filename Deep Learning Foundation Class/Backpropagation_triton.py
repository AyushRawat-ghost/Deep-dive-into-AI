import triton
import triton.language as tl
import torch

@triton.jit
def perceptron_backward_kernel(
    dz_ptr, x_ptr, dw_ptr, db_ptr,
    M, N, K,
    stride_dzm, stride_dzn,
    stride_xm, stride_xk,
    stride_dwn, stride_dwk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)  # Out features index
    pid_k = tl.program_id(axis=1)  # In features index

    cols_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    feats_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    col_mask = cols_offsets < N
    feat_mask = feats_offsets < K

    dw_accumulator = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    db_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, tl.cdiv(M, BLOCK_M)):
        m_offsets = m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        dz_tile = tl.load(
            dz_ptr + (m_offsets[:, None] * stride_dzm + cols_offsets[None, :] * stride_dzn),
            mask=m_mask[:, None] & col_mask[None, :], other=0.0
        )
        
        x_tile = tl.load(
            x_ptr + (m_offsets[:, None] * stride_xm + feats_offsets[None, :] * stride_xk),
            mask=m_mask[:, None] & feat_mask[None, :], other=0.0
        )

        dw_accumulator += tl.dot(tl.trans(dz_tile), x_tile)
        
        if pid_k == 0:
            db_accumulator += tl.sum(dz_tile, axis=0)

    dw_write_ptr = dw_ptr + (cols_offsets[:, None] * stride_dwn + feats_offsets[None, :] * stride_dwk)
    tl.store(dw_write_ptr, dw_accumulator, mask=col_mask[:, None] & feat_mask[None, :])
    
    if pid_k == 0:
        tl.store(db_ptr + cols_offsets, db_accumulator, mask=col_mask)
    

def run_triton_backward(dZ : torch.tensor,X:torch.Tensor):
    M,N = dZ.shape
    _,K = X.shape

    dW = torch.empty((N,K),device='cuda',dtype=torch.float32)
    db = torch.empty((N,),device='cuda',dtype=torch.float32)
    
    BLOCK_M,BLOCK_N,BLOCK_K = 16,16,16
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    
    perceptron_backward_kernel[grid](
        dZ, X, dW, db,
        M, N, K,
        dZ.stride(0), dZ.stride(1),
        X.stride(0), X.stride(1),
        dW.stride(0), dW.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return dW, db
    
    