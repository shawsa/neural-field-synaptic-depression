from cmath import exp, pi

def is_pow2(N: int):
    if N < 0:
        N = -N
    pow2 = 1
    while N > pow2:
        pow2 = pow2 << 1
    return N == pow2

def dct(x: list) -> list:
    N = len(x)
    assert is_pow2(N)
    assert N > 0
    w = exp(-2j*pi/N)
    return [sum(w**(n*k)*val for k, val in enumerate(x))
            for n in range(N)]


def fft(x: list) -> list:
    N = len(x)
    assert is_pow2(N)
    assert N > 0
    if N == 1:
        return x
    w = exp(-2j*pi/N)
    even = fft(x[::2])
    odd = [w**k * freq for k, freq in enumerate(fft(x[1::2]))]
    return [e + o for e, o in zip(even, odd)] + \
           [e - o for e, o in zip(even, odd)]

if __name__ == '__main__':
    from numpy.fft import fft as nfft
    from time import perf_counter
    x = [1.0 for _ in range(4)] + [0.0 for _ in range(4)]
    x_hat = fft(x)
    print([abs(freq) for freq in x_hat])
    print(nfft(x))

    test = [1.0 for _ in range(2**12)]
    for name, method in [
                ('numpy', nfft),
                ('recursive', fft),
                ('direct', dct)
            ]:
        print(f'{name:>10}: ', end='')
        tic = perf_counter()
        _ = method(test)
        toc = perf_counter()
        print(f'{toc-tic:e}s')
