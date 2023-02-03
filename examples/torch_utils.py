# https://www.linkedin.com/pulse/visual-studio-code-auto-format-shortcut-key-billour-ou/?trk=pulse-article_more-articles_related-content-card
def add_params_grads(params_dict, grads_dict):
    r""" Calculate surrogate term by dot product of
        the previous diff_gradients and current local parameters
        Args:
            params_dict: {'state_dict': param.data} current model parameters value
            grads_dict: {'state_dict': diff_grads} differentiate gradients between
                        the previous global model (weighted by \eta) and local model parameters
                        = dot ((\eta * \delta F (w^{t-1}) - F_n (w^{t-1}), w^t)
        Return:
            Tensor()
    """
    sum = 0.0
    for k in params_dict.keys():
        sum += (params_dict[k] * grads_dict[k]).sum()
    return sum


