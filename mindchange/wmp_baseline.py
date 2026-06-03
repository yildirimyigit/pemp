"""Wavelet Movement Primitive (WMP) baseline -- faithful to Zhang et al. 2025
(IEEE RA-L, 'Wavelet Movement Primitives: A Unified Framework for Learning
Discrete and Rhythmic Movements'), with the phase-adaptive weight adjustment
for rhythmic motions added.

Implemented (used by the FluidLab mixing comparison):
  * DWT-based wavelet coefficient encoding/decoding (Section II.A + III.B).
  * Probabilistic Gaussian model over wavelet coefficients (eq. 9).
  * Soft-thresholding denoising preprocessing (eq. 6, optional, default off).
  * Standard Gaussian via-point conditioning (eq. 5; used inside the rhythmic
    update as the "first pass" before the phase-adaptive correction).
  * Phase-adaptive weight adjustment for rhythmic motions (eqs. 14-17).
    Selected via predict/condition(method="phase_adaptive").  This is the
    paper's KEY contribution for rhythmic tasks.

Not implemented (not relevant to the rhythmic mixing comparison):
  * Local-frame WMP (LF-WMP, Section III.C.1, for discrete tasks).
  * Auto rhythmic/discrete change-point segmentation via DWT (Section III.A);
    we assume the input is purely rhythmic.

Our pragmatic extension (NOT in the paper):
  * Linear contextual prior  mu_w(g) = B^T [1; g]  so a single WMP can serve
    all frequencies in the mixing dataset; the paper trains one WMP per rhythm.

Dependency: pip install PyWavelets.
Data convention:
    Y: (n_demos, n_steps, n_dims)        # demonstrations
    contexts: (n_demos, context_dim)     # e.g. frequency g
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pywt
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyWavelets first: pip install PyWavelets") from exc


ArrayLike = Union[np.ndarray, Sequence[float]]


# ----------------------------- helpers ------------------------------------- #
def _as_3d(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2:
        Y = Y[None, :, :]
    if Y.ndim != 3:
        raise ValueError("Y must have shape (n_demos, n_steps, n_dims) or (n_steps, n_dims).")
    return Y


def _normalize_time(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if len(t) < 2:
        raise ValueError("Each time array must contain at least two samples.")
    denom = t[-1] - t[0]
    if abs(denom) < 1e-12:
        raise ValueError("Invalid timestamps: first and last timestamp are identical.")
    return (t - t[0]) / denom


def resample_trajectories(trajectories, times=None, n_steps=None):
    """Resample demonstrations onto a common normalized time grid."""
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3 and times is None:
        Y = trajectories.astype(float)
        if n_steps is None or n_steps == Y.shape[1]:
            return Y, np.linspace(0.0, 1.0, Y.shape[1])
    trajs = [np.asarray(y, dtype=float) for y in trajectories]
    if not trajs:
        raise ValueError("No trajectories provided.")
    dims = {y.shape[1] for y in trajs}
    if len(dims) != 1:
        raise ValueError("All trajectories must have the same output dimension.")
    if n_steps is None:
        n_steps = int(np.median([y.shape[0] for y in trajs]))
    grid = np.linspace(0.0, 1.0, n_steps)
    if times is None:
        time_list = [np.linspace(0.0, 1.0, y.shape[0]) for y in trajs]
    else:
        time_list = ([times] if (isinstance(times, np.ndarray) and times.ndim == 1
                                 and len(trajs) == 1) else list(times))
        if len(time_list) != len(trajs):
            raise ValueError("times must contain one timestamp array per trajectory.")
    Y = np.zeros((len(trajs), n_steps, trajs[0].shape[1]), dtype=float)
    for i, (y, t) in enumerate(zip(trajs, time_list)):
        tn = _normalize_time(np.asarray(t, dtype=float))
        if len(tn) != y.shape[0]:
            raise ValueError("Each timestamp array length must match its trajectory length.")
        for d in range(y.shape[1]):
            Y[i, :, d] = np.interp(grid, tn, y[:, d])
    return Y, grid


def _soft_threshold_denoise(y: np.ndarray, wavelet: str, mode: str, level: Optional[int]) -> np.ndarray:
    """Universal-threshold soft denoising on the detail coefficients (eq. 6)."""
    coeffs = pywt.wavedec(y, wavelet, mode=mode, level=level)
    if len(coeffs) < 2:
        return y
    detail0 = coeffs[-1]
    if detail0.size == 0:
        return y
    lam = (np.median(np.abs(detail0)) / 0.6745) * np.sqrt(2.0 * np.log(max(len(y), 2)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, lam, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(new_coeffs, wavelet, mode=mode)[: len(y)]


@dataclass
class WMPFit:
    mu_w: np.ndarray
    sigma_w: np.ndarray
    residual_sigma_w: Optional[np.ndarray] = None
    context_coef: Optional[np.ndarray] = None
    W_train: Optional[np.ndarray] = None         # (n_demos, K)  per-demo coefficients
    Y_train: Optional[np.ndarray] = None         # (n_demos, n_steps, n_dims)  resampled demos
    C_train: Optional[np.ndarray] = None         # (n_demos, ctx_dim) contexts (if any)


# ============================================================================ #
class WaveletMovementPrimitive:
    """WMP with both standard via-point conditioning and the paper's
    phase-adaptive weight adjustment (eqs. 14-17) for rhythmic tasks."""

    def __init__(self, wavelet="db4", level=None, mode="periodization",
                 reg=1e-6, obs_noise=1e-5, ridge=1e-6, denoise=False):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.reg = float(reg)
        self.obs_noise = float(obs_noise)
        self.ridge = float(ridge)
        self.denoise = bool(denoise)

        self.n_steps: Optional[int] = None
        self.n_dims: Optional[int] = None
        self.grid: Optional[np.ndarray] = None
        self.coeff_slices = None
        self.coeff_shape = None
        self.k_per_dim: Optional[int] = None
        self.Phi: Optional[np.ndarray] = None   # (T, K_per_dim) synthesis basis per dim
        self.fit_: Optional[WMPFit] = None

    # ------------- wavelet plumbing ------------------------------------------ #
    def _check_fitted(self):
        if self.fit_ is None:
            raise RuntimeError("Call fit(...) before using this method.")

    def _setup_wavelet_geometry(self, n_steps):
        zero = np.zeros(n_steps, dtype=float)
        coeffs = pywt.wavedec(zero, self.wavelet, mode=self.mode, level=self.level)
        arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        self.coeff_shape = arr.shape
        self.k_per_dim = arr.size
        Phi = np.zeros((n_steps, self.k_per_dim), dtype=float)
        for k in range(self.k_per_dim):
            unit = np.zeros(self.k_per_dim, dtype=float); unit[k] = 1.0
            coeffs_k = pywt.array_to_coeffs(unit.reshape(self.coeff_shape),
                                            self.coeff_slices, output_format="wavedec")
            Phi[:, k] = pywt.waverec(coeffs_k, self.wavelet, mode=self.mode)[:n_steps]
        self.Phi = Phi

    def _encode_one_dim(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, mode=self.mode, level=self.level)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.reshape(-1)

    def _decode_one_dim(self, w):
        coeffs = pywt.array_to_coeffs(w.reshape(self.coeff_shape),
                                       self.coeff_slices, output_format="wavedec")
        return pywt.waverec(coeffs, self.wavelet, mode=self.mode)[: self.n_steps]

    def encode(self, Y):
        Y = _as_3d(Y)
        if self.n_steps is None:
            self.n_steps, self.n_dims = Y.shape[1], Y.shape[2]
            self._setup_wavelet_geometry(self.n_steps)
        if Y.shape[1] != self.n_steps or Y.shape[2] != self.n_dims:
            raise ValueError("Y shape does not match fitted WMP dimensions.")
        W = np.zeros((Y.shape[0], self.n_dims * self.k_per_dim), dtype=float)
        for n in range(Y.shape[0]):
            W[n] = np.concatenate([self._encode_one_dim(Y[n, :, d]) for d in range(self.n_dims)])
        return W

    def decode(self, w):
        self._check_fitted()
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.size != self.n_dims * self.k_per_dim:
            raise ValueError(f"Expected coefficient vector of length {self.n_dims * self.k_per_dim}, got {w.size}.")
        Y = np.zeros((self.n_steps, self.n_dims), dtype=float)
        for d in range(self.n_dims):
            lo, hi = d * self.k_per_dim, (d + 1) * self.k_per_dim
            Y[:, d] = self._decode_one_dim(w[lo:hi])
        return Y

    # ------------- fit ------------------------------------------------------- #
    def fit(self, trajectories, times=None, n_steps=None, contexts=None):
        Y, grid = resample_trajectories(trajectories, times=times, n_steps=n_steps)
        if self.denoise:
            # eq. 6: per-demo, per-dim universal-threshold soft denoising
            Y = np.stack([
                np.stack([_soft_threshold_denoise(Y[n, :, d], self.wavelet, self.mode, self.level)
                          for d in range(Y.shape[2])], axis=-1)
                for n in range(Y.shape[0])
            ], axis=0)
        self.grid = grid
        self.n_steps, self.n_dims = Y.shape[1], Y.shape[2]
        self._setup_wavelet_geometry(self.n_steps)
        W = self.encode(Y)

        mu = W.mean(axis=0)
        sigma = np.cov(W, rowvar=False) if W.shape[0] > 1 else np.eye(W.shape[1]) * self.reg
        sigma = np.atleast_2d(sigma) + self.reg * np.eye(W.shape[1])

        context_coef = None; residual_sigma = None; C = None
        if contexts is not None:
            C = np.asarray(contexts, dtype=float)
            if C.ndim == 1:
                C = C[:, None]
            if C.shape[0] != W.shape[0]:
                raise ValueError("contexts must have one row per demonstration.")
            X = np.concatenate([np.ones((C.shape[0], 1)), C], axis=1)
            A = X.T @ X + self.ridge * np.eye(X.shape[1])
            B = np.linalg.solve(A, X.T @ W)
            R = W - X @ B
            residual_sigma = np.cov(R, rowvar=False) if W.shape[0] > 1 else np.eye(W.shape[1]) * self.reg
            residual_sigma = np.atleast_2d(residual_sigma) + self.reg * np.eye(W.shape[1])
            context_coef = B

        self.fit_ = WMPFit(mu_w=mu, sigma_w=sigma,
                           residual_sigma_w=residual_sigma, context_coef=context_coef,
                           W_train=W, Y_train=Y, C_train=C)
        return self

    # ------------- contextual prior & basis utilities ------------------------ #
    def _prior_for_context(self, context):
        self._check_fitted()
        if context is None or self.fit_.context_coef is None:
            return self.fit_.mu_w.copy(), self.fit_.sigma_w.copy()
        c = np.asarray(context, dtype=float).reshape(1, -1)
        x = np.concatenate([np.ones((1, 1)), c], axis=1)
        if x.shape[1] != self.fit_.context_coef.shape[0]:
            raise ValueError("context dimension does not match training contexts.")
        return (x @ self.fit_.context_coef).reshape(-1), self.fit_.residual_sigma_w.copy()

    def _basis_row(self, t_norm):
        t = float(np.clip(t_norm, 0.0, 1.0))
        x = t * (self.n_steps - 1)
        i0 = int(np.floor(x)); i1 = min(i0 + 1, self.n_steps - 1); a = x - i0
        return (1.0 - a) * self.Phi[i0] + a * self.Phi[i1]

    def _observation_matrix(self, t, dims=None):
        t = np.asarray(t, dtype=float).reshape(-1)
        if dims is None:
            dims = list(range(self.n_dims))
        rows = []
        total_k = self.n_dims * self.k_per_dim
        for tm in t:
            phi = self._basis_row(tm)
            for d in dims:
                row = np.zeros(total_k, dtype=float)
                row[d * self.k_per_dim : (d + 1) * self.k_per_dim] = phi
                rows.append(row)
        return np.vstack(rows)

    def _eval_demos_at_time(self, t_norm):
        """Interpolate each stored demonstration trajectory at normalized time t in [0,1].
        Returns (n_demos, n_dims)."""
        if self.fit_.Y_train is None:
            raise RuntimeError("Y_train not stored.")
        t = float(np.clip(t_norm, 0.0, 1.0))
        x = t * (self.n_steps - 1)
        i0 = int(np.floor(x)); i1 = min(i0 + 1, self.n_steps - 1); a = x - i0
        return (1.0 - a) * self.fit_.Y_train[:, i0, :] + a * self.fit_.Y_train[:, i1, :]

    # ------------- standard Gaussian conditioning (eq. 5) -------------------- #
    def condition_gaussian(self, t, y, context=None, dims=None, obs_noise=None,
                           init_mu=None, init_Sigma=None):
        if init_mu is None or init_Sigma is None:
            mu, Sigma = self._prior_for_context(context)
            if init_mu is not None: mu = init_mu
            if init_Sigma is not None: Sigma = init_Sigma
        else:
            mu, Sigma = init_mu.copy(), init_Sigma.copy()
        t = np.asarray(t, dtype=float).reshape(-1)
        if dims is None:
            dims = list(range(self.n_dims))
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(len(t), len(dims))
        if y.shape != (len(t), len(dims)):
            raise ValueError(f"y must have shape {(len(t), len(dims))}, got {y.shape}.")
        H = self._observation_matrix(t, dims=dims)
        noise = self.obs_noise if obs_noise is None else float(obs_noise)
        R = (noise ** 2) * np.eye(H.shape[0])
        S = H @ Sigma @ H.T + R
        K = Sigma @ H.T @ np.linalg.pinv(S)
        mu_post = mu + K @ (y.reshape(-1) - H @ mu)
        Sigma_post = Sigma - K @ H @ Sigma
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T) + self.reg * np.eye(Sigma_post.shape[0])
        return mu_post, Sigma_post

    # ------------- phase-adaptive update (eqs. 14-17) ------------------------ #
    def phase_adaptive_update(self, t_obs, y_obs, context=None, dims=None,
                              sigma=0.3, beta=1.0, dt=1.0,
                              init_mu=None, init_Sigma=None,
                              do_gaussian_first=True, batched=True):
        """Phase-adaptive weight adjustment for rhythmic tasks (Section III.C.2).

        For each observation (t_k, y_k):
          1. Compute l_i = ||y_k - y_demo_i(t_k)||  (one per demo).
          2. Normalize l_i to l_bar_i in [0,1] and form RBF weights
                alpha_i = exp(-l_bar_i^2 / (2 sigma^2))           (eqs. 14-15)
          3. Weighted target over per-demo wavelet coefficients c_i = W_train[i]:
                mu*  = sum_i alpha_i c_i / sum_i alpha_i
                Sig* = sum_i alpha_i (c_i - mu*)(c_i - mu*)^T / sum_i alpha_i  (eq. 16)
          4. Kalman blend toward the target (eq. 17):
                Sigma_{t+1} = (Sigma_t^-1 + Sig*^-1)^-1
                K           = Sigma_{t+1} Sig*^-1
                mu_{t+1}    = mu_t + dt * beta * K (mu* - mu_t)

        do_gaussian_first=True applies the standard via-point conditioning of
        eq. 5 BEFORE the phase-adaptive step (as in the paper's narrative:
        "We still use the WMPs to learn the demonstrations and apply conditional
        probability to obtain the posterior c_t ~ N(mu_t, Sigma_t)").

        batched=True (default) does ONE Kalman update with the mean per-demo
        weight averaged across all observations -- O(K^3) instead of O(n_obs*K^3).
        Set batched=False for the paper-strict per-observation iteration; useful
        for misaligned-phase demos, near-equivalent when phases are aligned.
        """
        self._check_fitted()
        if self.fit_.W_train is None or self.fit_.Y_train is None:
            raise RuntimeError("W_train/Y_train not stored; re-fit the model.")

        t_obs = np.atleast_1d(np.asarray(t_obs, dtype=float))
        y_obs = np.asarray(y_obs, dtype=float)
        if dims is None:
            dims = list(range(self.n_dims))
        if y_obs.ndim == 1:
            y_obs = y_obs.reshape(len(t_obs), len(dims))
        if y_obs.shape != (len(t_obs), len(dims)):
            raise ValueError(f"y_obs must have shape {(len(t_obs), len(dims))}, got {y_obs.shape}.")

        # 0. start from the (possibly context-dependent) prior, optionally do
        #    the standard via-point conditioning as the "first pass"
        if init_mu is None or init_Sigma is None:
            mu, Sigma = self._prior_for_context(context)
            if do_gaussian_first:
                mu, Sigma = self.condition_gaussian(t_obs, y_obs, context=context, dims=dims,
                                                    init_mu=mu, init_Sigma=Sigma)
            if init_mu is not None: mu = init_mu
            if init_Sigma is not None: Sigma = init_Sigma
        else:
            mu, Sigma = init_mu.copy(), init_Sigma.copy()

        W = self.fit_.W_train                                  # (n_demos, K)
        K_dim = W.shape[1]
        eye_K = np.eye(K_dim)
        twosig2 = 2.0 * (sigma ** 2) + 1e-12

        # Per-observation per-demo weight (normalized at each obs so each obs
        # contributes equally).  Then either iterate (paper-strict, one Kalman
        # update per obs) or aggregate weights across obs and do ONE update
        # (default; far faster, exact-equivalent up to first order on aligned
        # data, where Gaussian conditioning has already done the heavy lifting).
        alphas = np.zeros((len(t_obs), W.shape[0]), dtype=float)
        for k, (tk, yk) in enumerate(zip(t_obs, y_obs)):
            y_demos = self._eval_demos_at_time(tk)             # (n_demos, n_dims)
            l = np.linalg.norm(y_demos[:, dims] - yk[None, :], axis=1)
            lo, hi = float(l.min()), float(l.max())
            l_bar = np.zeros_like(l) if (hi - lo) < 1e-12 else (l - lo) / (hi - lo)
            a = np.exp(-(l_bar ** 2) / twosig2)
            s = float(a.sum())
            alphas[k] = a / s if s > 1e-12 else 0.0

        def _kalman_step(mu, Sigma, an):
            mu_star = (an[:, None] * W).sum(axis=0)
            D = W - mu_star[None, :]
            Sigma_star = np.einsum("i,ij,ik->jk", an, D, D) + self.reg * eye_K
            try:
                Sigma_inv = np.linalg.inv(Sigma + self.reg * eye_K)
                Star_inv = np.linalg.inv(Sigma_star)
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(Sigma); Star_inv = np.linalg.pinv(Sigma_star)
            Sigma_new = np.linalg.inv(Sigma_inv + Star_inv)
            Kgain = Sigma_new @ Star_inv
            mu_new = mu + dt * beta * (Kgain @ (mu_star - mu))
            return mu_new, 0.5 * (Sigma_new + Sigma_new.T)

        if batched:
            # mean over obs of the per-obs normalized weights -> one update
            an = alphas.mean(axis=0)
            s = float(an.sum())
            if s >= 1e-12:
                an = an / s
                mu, Sigma = _kalman_step(mu, Sigma, an)
        else:
            for an in alphas:
                if float(an.sum()) < 1e-12:
                    continue
                mu, Sigma = _kalman_step(mu, Sigma, an)

        return mu, Sigma

    # ------------- predict (gaussian or phase_adaptive) ---------------------- #
    def condition(self, t, y, context=None, dims=None, obs_noise=None,
                  method="gaussian", **method_kwargs):
        """Posterior over wavelet coefficients given observation(s).

        method="gaussian"        : standard via-point conditioning (eq. 5).
        method="phase_adaptive"  : paper's eqs. 14-17 (for rhythmic motions).
                                   Forwards sigma, beta, dt, do_gaussian_first
                                   via **method_kwargs.
        """
        if method == "gaussian":
            return self.condition_gaussian(t, y, context=context, dims=dims, obs_noise=obs_noise)
        if method == "phase_adaptive":
            return self.phase_adaptive_update(t, y, context=context, dims=dims, **method_kwargs)
        raise ValueError(f"unknown method {method!r}; use 'gaussian' or 'phase_adaptive'.")

    def predict(self, context=None, t_cond=None, y_cond=None, dims=None,
                method="gaussian", return_std=False, **method_kwargs):
        if t_cond is None:
            mu, Sigma = self._prior_for_context(context)
        else:
            if y_cond is None:
                raise ValueError("y_cond is required when t_cond is provided.")
            mu, Sigma = self.condition(t_cond, y_cond, context=context, dims=dims,
                                       method=method, **method_kwargs)
        y_mean = self.decode(mu)
        if not return_std:
            return y_mean
        std = np.zeros_like(y_mean)
        for i, tm in enumerate(self.grid):
            H = self._observation_matrix([tm])
            std[i] = np.sqrt(np.maximum(np.diag(H @ Sigma @ H.T), 0.0))
        return y_mean, std

    def sample(self, n=1, context=None, t_cond=None, y_cond=None, dims=None,
               method="gaussian", random_state=None, **method_kwargs):
        rng = np.random.default_rng(random_state)
        if t_cond is None:
            mu, Sigma = self._prior_for_context(context)
        else:
            mu, Sigma = self.condition(t_cond, y_cond, context=context, dims=dims,
                                       method=method, **method_kwargs)
        W = rng.multivariate_normal(mu, Sigma, size=n)
        return np.stack([self.decode(w) for w in W], axis=0)

    @staticmethod
    def mse(Y_true, Y_pred):
        return float(np.mean((np.asarray(Y_true, dtype=float) - np.asarray(Y_pred, dtype=float)) ** 2))


# ============================================================================ #
if __name__ == "__main__":
    # Sanity: rhythmic sinusoids with different frequencies + small phase noise.
    rng = np.random.default_rng(0)
    n_demos, T, D = 30, 200, 1
    grid = np.linspace(0.0, 1.0, T)
    freqs = rng.uniform(2.0, 5.0, size=n_demos)
    phases = rng.uniform(0.0, 0.4 * np.pi, size=n_demos)  # mild misalignment
    Y = np.stack([np.sin(2.0 * np.pi * f * grid + p)[:, None]
                  for f, p in zip(freqs, phases)], axis=0)
    m = WaveletMovementPrimitive(wavelet="db4", obs_noise=1e-3).fit(Y, contexts=freqs[:, None])
    pred_g = m.predict(context=[3.0], t_cond=[0.0], y_cond=[[0.0]], method="gaussian")
    pred_p = m.predict(context=[3.0], t_cond=[0.0], y_cond=[[0.0]], method="phase_adaptive")
    print("gaussian pred shape:", pred_g.shape, "  phase_adaptive pred shape:", pred_p.shape)
    print("their MSE difference (smaller=more similar):", m.mse(pred_g, pred_p))
