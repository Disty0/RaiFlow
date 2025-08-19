import torch
import math

# Modified from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

@torch.no_grad()
def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]

    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :].mul_(math.pi / (2 * N))
    W_r = torch.cos(k)
    n_W_i = -torch.sin(k)

    V = torch.addcmul((Vc[:, :, 0] * W_r), Vc[:, :, 1], n_W_i)
    if norm == 'ortho':
        V[:, 0].mul_(0.5 / math.sqrt(N))
        V[:, 1:].mul_(0.5 / math.sqrt(N / 2))

    V = V.view(x_shape).mul_(2)
    return V


@torch.no_grad()
def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N).div_(2)
    if norm == 'ortho':
        X_v[:, 0].mul_(math.sqrt(N) * 2)
        X_v[:, 1:].mul_(math.sqrt(N / 2) * 2)

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :].mul_(math.pi / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_i = torch.cat([X_v.new_zeros((X_v.shape[0], 1)), -(X_v.flip([1])[:, :-1])], dim=1)
    V_r = torch.addcmul((X_v * W_r), V_t_i, -W_i)
    V_i = torch.addcmul((X_v * W_i), V_t_i, W_r)

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2].add_(v[:, :N - (N // 2)])
    x[:, 1::2].add_(v.flip([1])[:, :N // 2])

    x = x.view(x_shape)
    return x


@torch.no_grad()
def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm).transpose_(-1, -2)
    X2 = dct(X1, norm=norm).transpose_(-1, -2)
    return X2


@torch.no_grad()
def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm).transpose_(-1, -2)
    x2 = idct(x1, norm=norm).transpose_(-1, -2)
    return x2
