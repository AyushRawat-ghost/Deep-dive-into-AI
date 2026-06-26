import torch
import triton
import triton.language as tl

@triton.jit
def single_perceptron_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(axis=0)
    if pid == 0:
        offsets = tl.arange(0,BLOCK_SIZE)
        mask = offsets < n_elements
        
        x=tl.load(x_ptr + offsets,mask=mask)
        w=tl.load(w_ptr+offsets,mask=mask)
        bias=tl.load(b_ptr)

        linear_combination = tl.sum(x*w,axis=0)+bias
        binary_output = tl.where(linear_combination>=.0,1,0)

        tl.store(out_ptr,binary_output)


def run_single_perceptron(
    x:torch.Tensor,
    w:torch.Tensor,
    b:torch.Tensor,
):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All inputs must be CUDA Tensors"
    n_elements = x.numel()
    output = torch.empty((1,),device=x.device,dtype=torch.int32)
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    grid = (1,)
    single_perceptron_kernel[grid](
        x,
        w,
        b,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output.item()
    

    
    