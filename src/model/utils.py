def get_conv_padding_size(kernel, stride, dillation):
    """
    Counts padding for Conv1d based on kernel, stride, and dillation
    So the length of output of Conv layer is the same to input one 
    """
    if stride == 1:
        # (final - 1) = input + 2 * pad - dil * (kernel - 1) - 1, final == input
        # 2 * pad = dil * (kernel - 1)
        # pad = dil * (kernel - 1) / 2
        return dillation * (kernel - 1) // 2
    else:
        raise NotImplementedError()
