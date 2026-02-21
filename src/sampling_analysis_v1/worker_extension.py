"""vLLM Worker Extension for weight perturbation sampling.

Stores base weights in CPU memory and applies perturbations by resetting
to base + noise each time. Zero numerical drift across perturbations.
"""

import torch


class SamplingWorkerExtension:

    def store_base_weights(self):
        """Save current model weights to CPU memory as the reference base."""
        self._base_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self._base_weights[name] = p.data.clone().cpu()
        torch.cuda.synchronize()
        return True

    def apply_perturbation(self, seed, sigma):
        """Reset to base weights then add Gaussian perturbation."""
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name].to(p.device))
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(float(sigma) * noise)
            del noise
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_base_weights(self):
        """Restore model to stored base weights."""
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name].to(p.device))
        torch.cuda.synchronize()
        return True
