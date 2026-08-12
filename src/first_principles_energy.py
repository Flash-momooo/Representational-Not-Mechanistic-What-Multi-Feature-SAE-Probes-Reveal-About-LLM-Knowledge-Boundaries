"""A self-contained first-principles contextual consistency energy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class EnergyFitDiagnostics:
    iterations: int
    relative_residual: float
    weight_norm: float


class FirstPrinciplesConsistencyEnergy:
    """Fuse commitment impulse and confidence into one candidate risk energy.

    The module enforces three structural constraints:

    - subtraction of the pre-commitment state removes common context shifts;
    - adjacent yes/no training pairs induce an antisymmetric learning signal;
    - nuisance contexts enter as a zero-energy quadratic penalty.

    No classifier or external fusion implementation is used. The energy
    direction is obtained with an internal conjugate-gradient solver.
    """

    def __init__(
        self,
        ridge: float = 0.10,
        intervention_strength: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-7,
        device: str | None = None,
    ) -> None:
        self.ridge = float(ridge)
        self.intervention_strength = float(intervention_strength)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_mean: np.ndarray | None = None
        self.hidden_rms: np.ndarray | None = None
        self.surprisal_mean: np.ndarray | None = None
        self.surprisal_rms: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.diagnostics: EnergyFitDiagnostics | None = None

    @staticmethod
    def _validate_inputs(
        committed: np.ndarray,
        pre_commitment: np.ndarray,
        selected_probability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        committed = np.asarray(committed, dtype=np.float32)
        pre_commitment = np.asarray(pre_commitment, dtype=np.float32)
        probability = np.asarray(selected_probability, dtype=np.float32).reshape(-1, 1)
        if committed.shape != pre_commitment.shape or committed.ndim != 2:
            raise ValueError("Committed and pre-commitment states must share a 2D shape")
        if len(probability) != len(committed):
            raise ValueError("One selected probability is required per candidate state")
        return committed, pre_commitment, np.clip(probability, 1e-7, 1.0)

    @staticmethod
    def _primitive_views(
        committed: np.ndarray,
        pre_commitment: np.ndarray,
        probability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        impulse = committed - pre_commitment
        susceptibility = 4.0 * probability * (1.0 - probability)
        hidden = susceptibility * impulse
        surprisal = -np.log(probability)
        return hidden.astype(np.float32), surprisal.astype(np.float32)

    @staticmethod
    def _pair_difference(features: np.ndarray) -> np.ndarray:
        return features[0::2] - features[1::2]

    def _fit_scaler(self, hidden: np.ndarray, surprisal: np.ndarray) -> None:
        self.hidden_mean = hidden.mean(axis=0, keepdims=True)
        self.hidden_rms = np.maximum(
            np.sqrt(np.mean((hidden - self.hidden_mean) ** 2, axis=0, keepdims=True)),
            1e-5,
        )
        self.surprisal_mean = surprisal.mean(axis=0, keepdims=True)
        self.surprisal_rms = np.maximum(
            np.sqrt(np.mean(
                (surprisal - self.surprisal_mean) ** 2, axis=0, keepdims=True
            )),
            1e-5,
        )

    def _transform(self, hidden: np.ndarray, surprisal: np.ndarray) -> np.ndarray:
        if any(value is None for value in (
            self.hidden_mean, self.hidden_rms,
            self.surprisal_mean, self.surprisal_rms,
        )):
            raise RuntimeError("The energy scaler has not been fitted")
        hidden_block = (hidden - self.hidden_mean) / self.hidden_rms
        hidden_block = hidden_block / np.sqrt(hidden.shape[1])
        surprisal_block = (surprisal - self.surprisal_mean) / self.surprisal_rms
        return np.concatenate((hidden_block, surprisal_block), axis=1).astype(np.float32)

    def _solve(
        self,
        signal: np.ndarray,
        signs: np.ndarray,
        controls: list[np.ndarray],
    ) -> np.ndarray:
        x = torch.as_tensor(signal, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(signs, dtype=torch.float32, device=self.device)
        nuisance = [
            torch.as_tensor(control, dtype=torch.float32, device=self.device)
            for control in controls
        ]
        moment = x.T @ y / len(x)

        def matvec(vector: torch.Tensor) -> torch.Tensor:
            output = self.ridge * vector + x.T @ (x @ vector) / len(x)
            for control in nuisance:
                output = output + (
                    self.intervention_strength
                    * control.T @ (control @ vector)
                    / len(control)
                )
            return output

        weights = torch.zeros_like(moment)
        residual = moment.clone()
        direction = residual.clone()
        initial_norm = torch.linalg.vector_norm(residual).clamp_min(1e-20)
        residual_sq = torch.dot(residual, residual)
        relative_residual = 1.0
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            response = matvec(direction)
            step = residual_sq / torch.dot(direction, response).clamp_min(1e-20)
            weights = weights + step * direction
            residual = residual - step * response
            new_residual_sq = torch.dot(residual, residual)
            relative_residual = float(torch.sqrt(new_residual_sq) / initial_norm)
            if relative_residual <= self.tolerance:
                break
            direction = residual + (new_residual_sq / residual_sq.clamp_min(1e-20)) * direction
            residual_sq = new_residual_sq

        self.diagnostics = EnergyFitDiagnostics(
            iterations=iterations,
            relative_residual=relative_residual,
            weight_norm=float(torch.linalg.vector_norm(weights)),
        )
        return weights.cpu().numpy()

    def fit(
        self,
        committed: np.ndarray,
        pre_commitment: np.ndarray,
        selected_probability: np.ndarray,
        pair_signs: np.ndarray,
        interventions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> "FirstPrinciplesConsistencyEnergy":
        """Fit from adjacent candidate pairs and optional nuisance contexts.

        `pair_signs` is +1 when the first candidate should have higher risk and
        -1 when the second candidate should have higher risk. Intervention
        tuples contain committed state, pre-commitment state, and selected
        probability in the same candidate order; they require no labels.
        """
        committed, pre_commitment, probability = self._validate_inputs(
            committed, pre_commitment, selected_probability
        )
        signs = np.asarray(pair_signs, dtype=np.float32).reshape(-1)
        if len(committed) % 2 or len(signs) * 2 != len(committed) or not np.all(
            np.isin(signs, (-1, 1))
        ):
            raise ValueError("pair_signs must contain one -1/+1 value per pair")

        hidden, surprisal = self._primitive_views(
            committed, pre_commitment, probability
        )
        self._fit_scaler(hidden, surprisal)
        signal = self._pair_difference(self._transform(hidden, surprisal))

        control_differences = []
        for intervention in interventions or []:
            control_c, control_pre, control_p = self._validate_inputs(*intervention)
            control_hidden, control_surprisal = self._primitive_views(
                control_c, control_pre, control_p
            )
            control_differences.append(self._pair_difference(
                self._transform(control_hidden, control_surprisal)
            ))
        self.weights = self._solve(signal, signs, control_differences)
        return self

    def score(
        self,
        committed: np.ndarray,
        pre_commitment: np.ndarray,
        selected_probability: np.ndarray,
    ) -> np.ndarray:
        """Return one scalar inconsistency energy per candidate."""
        if self.weights is None:
            raise RuntimeError("The energy module has not been fitted")
        committed, pre_commitment, probability = self._validate_inputs(
            committed, pre_commitment, selected_probability
        )
        hidden, surprisal = self._primitive_views(
            committed, pre_commitment, probability
        )
        return self._transform(hidden, surprisal) @ self.weights

    def pair_margin(
        self,
        committed: np.ndarray,
        pre_commitment: np.ndarray,
        selected_probability: np.ndarray,
    ) -> np.ndarray:
        """Return first-candidate energy minus second-candidate energy."""
        return self._pair_difference(
            self.score(committed, pre_commitment, selected_probability)
        )

    def state_dict(self) -> dict[str, np.ndarray | float | int]:
        if self.weights is None:
            raise RuntimeError("The energy module has not been fitted")
        return {
            "ridge": self.ridge,
            "intervention_strength": self.intervention_strength,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "hidden_mean": self.hidden_mean,
            "hidden_rms": self.hidden_rms,
            "surprisal_mean": self.surprisal_mean,
            "surprisal_rms": self.surprisal_rms,
            "weights": self.weights,
        }

    @classmethod
    def from_npz(
        cls, path: str | Path, device: str | None = None
    ) -> "FirstPrinciplesConsistencyEnergy":
        """Restore a serialized module without refitting."""
        state = np.load(path, allow_pickle=False)
        module = cls(
            ridge=float(state["ridge"]),
            intervention_strength=float(state["intervention_strength"]),
            max_iter=int(state["max_iter"]),
            tolerance=float(state["tolerance"]),
            device=device,
        )
        for name in (
            "hidden_mean", "hidden_rms", "surprisal_mean",
            "surprisal_rms", "weights",
        ):
            setattr(module, name, state[name].astype(np.float32))
        return module
