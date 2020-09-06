DEFAULT_SAMPLE_RATE_KHZ = 16

def windows(tensor, wsize=None, overlap = 50, step = None):
    window_size = wsize
    step_size = int(window_size // (1 // (overlap / 100)))
    return tensor.unfold(1, window_size, step or step_size)