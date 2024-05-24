import scipy.stats as stats
import numpy as np
import torch

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)
def sample_t(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t
if __name__ == '__main__':
    # Create an instance of the class
    exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')

    num_samples = 1000
    a = 4
    samples = sample_t(exponential_distribution, num_samples, a).numpy()

    # Plot the histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True)
    plt.savefig('exponential_samples.png', dpi=300)

