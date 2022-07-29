import logging
from numbers import Complex, Real
import numpy as np
from pade.minimization import lspparams, optimize_pparams

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PadeModel:
    def __init__(self, pparams=None):
        self.pparams = pparams
        if len(pparams) % 2:
            raise ValueError()

    def approx_num_den(self, z: Complex) -> Complex:
        """ Calculate the numerator and denominator of the Pade approximant """
        a = self.pparams[0 : len(self.pparams) // 2]
        b = self.pparams[len(self.pparams) // 2 :]
        r = len(a)
        num = 0
        with np.errstate(over="raise"):
            try:
                den = z**r
            except FloatingPointError as e:
                print(f"{z}**{r}")
                raise e
        for i, ai in enumerate(a):
            num += ai * z**i
        for i, bi in enumerate(b):
            den += bi * z**i
        return num, den

    def __call__(self, z: Complex) -> Complex:
        num, den = self.approx_num_den(z)
        return num / den

    def rho(self, w: Real, delta: Real = 0.01):
        """evaluate spectral function, delta Rydberg over the real line"""
        return -1 / np.pi * np.imag(self.__call__(w + delta * 1j))

    def physical(self, w, tol=0):
        for wi in w:
            if self.rho(wi) < tol:
                return False
        return True

    def plot_base(self, ax, z, s):
        ax[0].plot(np.imag(z), np.real(self(z)), label=f"se", color="#c33")
        ax[0].plot(np.imag(z), np.imag(self(z)), label=f"se", color="#933")
        ax[0].legend()
        ax[0].plot(np.imag(z), np.real(s), label=f"se", color="#3c3")
        ax[0].plot(np.imag(z), np.imag(s), label=f"se", color="#393")
        ax[0].legend()
        # get indexes where model is smaller than 1e4
        # plot like above but only points smaller than 1e4. Evaluate model, get indices, plot
        y = self(-z * 1j)
        idx = np.where(np.abs(y) < 1e4)
        ax[1].plot(np.imag(z[idx]), np.real(y[idx]), label=f"se", color="#c33", ms=2)
        ax[1].plot(np.imag(z[idx]), np.imag(y[idx]), label=f"se", color="#393", ms=2)
        ax[1].plot(
            np.imag(z[idx]), self.rho(np.imag(z[idx])), label=f"se", color="#3c3", ms=2
        )
        ax[1].legend()

    def plot(self, ax, z, s):
        self.plot_base(ax, z, s)

    @classmethod
    def from_specs(cls, z, s, M, N):
        z, s = np.array(z), np.array(s)
        samples_i = np.linspace(0, len(z) - 1, num=M, dtype=np.int)
        r = N // 2
        pparams = lspparams(r, z[samples_i], s[samples_i])
        return cls(pparams)


class AverageModel(PadeModel):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def __call__(self, z: Complex) -> Complex:
        res = 0
        for model, weight in zip(self.models, self.weights):
            res += model(z) * weight
        res /= np.sum(self.weights)
        return res

    @staticmethod
    def get_models(z, s, M, N, precise_iter=0):
        z, s, M, N = np.array(z), np.array(s), np.array(M), np.array(N)

        models = []
        for m in M:
            if m % 2:
                pass  # raise ValueError()
            for n in N:
                if n % 2:
                    raise ValueError()
                if n > m:
                    continue
                model = PadeModel.from_specs(z, s, m, n)
                if precise_iter:
                    pparams = optimize_pparams(model.pparams, z, s, precise_iter)
                    model = PadeModel(pparams)
                models.append(model)
        return models

        
class AutoWeightedModel(AverageModel):
    def __init__(self, models, w, get_weights=None):
        self.models = models
        if get_weights:
            self.get_weights=get_weights
        self.weights = self.get_weights(w)
    
    def get_weights(self, w):
        return NotImplementedError()

    @classmethod
    def from_specs(cls, z, s, M, N, w=None, precise_iter=0):
        """ Create an AverageLSModel on a list of models fitted on z and s with the specified ranges of M and N """

        if w is None:
            # make the assumption that w is the same as z rotated the real axis, as in qe
            w = np.imag(z)

        models = cls.get_models(z, s, M, N, precise_iter)
        return cls(models, w)


class AverageLSModel(AutoWeightedModel):
    def get_weights(self, w):
        return [model.physical(w) for model in self.models]
                

class AverageSimilarModel(AutoWeightedModel):
    def get_weights(self, w):
        weights = [self.Dc(i, w) for i in range(len(self.models))]
        if not any(weights):
            raise RuntimeError("No physical solutions")
        return weights

    def Dc(self, i, w):
        deviation = 0
        for j, model in enumerate(self.models):
            if j == i:
                continue
            if not model.physical(w):
                continue
            deviation += np.sum(np.abs(model.rho(w) - self.models[i].rho(w)))
        return deviation