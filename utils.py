DEFAULT_SAMPLE_RATE_KHZ = 16

def windows(tensor, window_size=None, overlap = 50, step = None):
    step_size = int(window_size // (1 // (overlap / 100)))

    return tensor.unfold(1, window_size, step or step_size)