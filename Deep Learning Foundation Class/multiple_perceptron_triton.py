import torch
import triton
import triton.language as tl

@triton.jit
def multiple_perceptron_kernel(
    x_ptr,w_ptr,b_ptr,z_ptr, # pointers for x, weights, bias, and output
    M,N,K,  # rows,hiiden_neurons,columns
    stride_xm,stride_xk, # tell how many integers to jump to get to the next row and col respectively
    stride_wn,stride_wk, 
    stride_zm,stride_zn,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_D : tl.constexpr,    
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_offset = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
    cols_offset = pid_n * BLOCK_N + tl.arange(0,BLOCK_N)

    row_mask = rows_offset<M
    col_mask = cols_offset<N
    accumaltor = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)

    for k in range(0,tl.cdiv(K,BLOCK_D)):
        k_offset = k * BLOCK_D + tl.arange(0,BLOCK_D)
        k_mask = k_offset<K

        x_tile = tl.load(
            x_ptr+(rows_offset[:,None]*stride_xm + k_offset[None,:]*stride_xk),
            mask=row_mask[:,None] & k_mask[None,:] , 
            other=0.0,
        )
        w_tile= tl.load(
            w_ptr+(cols_offset[None,:]*stride_wn + k_offset[:,None]*stride_wk),
            mask=col_mask[None,:] & k_mask[:,None] , 
            other=0.0,
        )
        accumaltor += tl.dot(x_tile,w_tile)
    
    # add bias 
    bias_tile = tl.load(
        b_ptr+cols_offset,
        mask=col_mask,
        other=0.0,
    )
    accumaltor += bias_tile[None,:]

    activated_output = tl.where(accumaltor>=.0,1.0,.0)
    output_write_ptr = z_ptr+(rows_offset[:,None]*stride_zm + cols_offset[None,:]*stride_zn)
    tl.store(output_write_ptr, activated_output, mask=row_mask[:, None] & col_mask[None, :])

def run_multiple_perceptron_layer(
    X: torch.Tensor, 
    W: torch.Tensor, 
    b: torch.Tensor
):
    assert X.is_cuda and W.is_cuda and b.is_cuda

    M, K = X.shape
    N, K = W.shape

    Z = torch.empty((M,N),device=X.device,dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    grid = (triton.cdiv(M,BLOCK_M),triton.cdiv(N,BLOCK_N))

    multiple_perceptron_kernel[grid](
        X, W, b, Z,
        M, N, K,
        X.stride(0),X.stride(1),
        W.stride(0),W.stride(1),
        Z.stride(0),Z.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_D=16,
        num_warps=4,
        num_stages=1,
    )
    return Z