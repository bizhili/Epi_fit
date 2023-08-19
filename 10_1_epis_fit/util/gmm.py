from sklearn.mixture import GaussianMixture
import numpy as np

def estimate_days(trueDistribution, timeHorizon, component, sample_size = 1000):
    propotionITrue= trueDistribution.cpu().detach().numpy()
    propotionITrue[propotionITrue < 0.001] = 0
    propotionITrue= propotionITrue/propotionITrue.sum()
    sampled_points = np.random.choice(timeHorizon, size=sample_size, p=propotionITrue)
    gmm = GaussianMixture(n_components= component)
    gmm.fit(sampled_points.reshape(-1, 1))
    return gmm.means_.flatten(), np.sqrt(gmm.covariances_).flatten(), gmm.weights_.flatten()