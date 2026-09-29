import torch
import math
from pathlib import Path
import fast_hadamard_transform

from utils.offload_utils import update_shared_offload_parameter


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if is_pow2(n):
        K = 1
        return hadK, K

    # load Hadamard matrices
    had_mat = torch.load(Path(__file__).parent / "had_mat.pt")
    k_list = list(had_mat.keys())
    k_list.sort()   # match smaller left-side Hadamard
    for k in k_list:
        if n % k == 0 and is_pow2(n // k):
            K, hadK = k, had_mat[k]
            hadK = hadK.T if transpose else hadK
            return hadK, K
        
    raise ValueError(f"Can't find appropriate Hadamard for size {n}!")


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def random_hadamard_matrix(size, device):
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    return matmul_hadU(Q).to(device)


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return fast_hadamard_transform.hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return fast_hadamard_transform.hadamard_transform(grad)


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadU_cuda(X, hadK, K):
    if X.numel() == 0:
        return X
    n = X.shape[-1]
    if K == 1:
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(n).sqrt()
    # if transpose:
    #     hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)
    input = HadamardTransform.apply(input.contiguous()) / torch.tensor(n).sqrt()
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None, sync=True):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)
    update_shared_offload_parameter(
        module, "weight", W_.to(device=dev, dtype=dtype), sync=sync
    )


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
