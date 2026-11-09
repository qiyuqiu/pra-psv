# Third-party sources

This anonymous artifact implements PRA and PSV. It does not claim authorship of the benchmark data, frozen language models or AttriGuard verification components.

- **AgentDojo:** https://github.com/ethz-spylab/agentdojo — executable environments, tool messages and evaluators. Obtain the MIT-licensed source and its copyright notice from upstream. This repository does not vendor AgentDojo.
- **DecomposedHarm / Monitoring Decomposition Attacks:** https://github.com/YuehHanChen/Monitoring-Decomposition-Attack — benchmark schemas and timely-detection conventions. The normalization and metric modules implement the release's documented conventions; original datasets are obtained separately under their upstream terms.
- **AttriGuard:** https://doi.org/10.5281/zenodo.20308739 — attenuation and fuzzy action survival. Obtain the official artifact and retain its notices. The upstream implementation is not bundled.
- **Qwen2.5-7B-Instruct:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct — external frozen encoder, revision `a09a35458c702b33eeacc393d103063234e8bc28`. Base-model weights are not distributed here.
- NumPy, PyTorch, Transformers and Accelerate retain their respective upstream licenses. They are installed as dependencies rather than vendored.

Public third-party author/project information is attribution, not identifying information about the authors of this anonymous submission.
