# MambaGram: Adaptive Time-Frequency Representations via Selective State Space Models

> **Status:** 🚧 Active research — pre-release. Paper in preparation for IEEE/ACM TASLP.

MambaGram is a learnable, input-dependent time-frequency (TF) representation built on selective state space models. It generalizes classical TF transforms — STFT, wavelet scalograms, gammatonegrams, and Mel-spectrograms — into a single unified, trainable front-end that adapts its time-frequency resolution to local signal content.

## Key Ideas

- **Unified TF representation.** We prove that STFT, scalograms, and gammatonegrams are special cases of MambaGram under specific parameter choices.
- **Input-dependent dynamics.** Via Mamba-style selectivity, MambaGram adapts window length and frequency resolution *per sample*, behaving spectrogram-like on harmonic signals and wavelet-like on transients.
- **Signal-processing grounded.** The layer is analyzed as a Gabor-like filter bank with learnable, input-dependent atoms — interpretable, not a black box.
- **End-to-end trainable.** MambaGram replaces the hand-crafted front-end in any audio classification pipeline.

## Repository Structure

```
mambagram/
├── src/mambagram/      # Core package: layers, models, utilities
├── experiments/        # Training scripts per dataset
├── notebooks/          # Exploration, sanity checks, visualizations
├── configs/            # YAML configs for reproducibility
├── tests/              # Unit tests
├── scripts/            # Shell scripts for batch experiments
├── paper/              # LaTeX source for the paper
├── figures/            # Final figures used in the paper
├── data/               # Local datasets (gitignored)
└── results/            # Checkpoints and logs (gitignored)
```

## Installation

Requires Python ≥ 3.10, PyTorch ≥ 2.1, and an NVIDIA GPU with CUDA for Mamba.

```bash
# Clone the repo
git clone https://github.com/<your-username>/mambagram.git
cd mambagram

# Create environment
python -m venv mambagram_env
source mambagram_env/bin/activate

# Install PyTorch (adjust CUDA version for your system)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
pip install -r requirements.txt

# Install Mamba (requires CUDA toolkit)
pip install mamba-ssm causal-conv1d --no-build-isolation

# Install the package in editable mode
pip install -e .
```

## Quick Start

*Coming soon — see `notebooks/01_mambagram_sanity_check.ipynb` once available.*

## Planned Experiments

- [ ] Synthetic signals (chirp, transient, harmonic) — visual sanity check
- [ ] ICBHI respiratory sound classification
- [ ] SpeechCommands V2 keyword spotting
- [ ] GTZAN / NSynth music classification
- [ ] ESC-50 environmental sound classification

## Citation

If you use this work, please cite (BibTeX to be added after arXiv posting):

```bibtex
@article{mambagram2026,
  title={MambaGram: Adaptive Time-Frequency Representations via Selective State Space Models},
  author={<Your Name> and <Advisor Name>},
  journal={In preparation},
  year={2026}
}
```

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of the excellent [mamba-ssm](https://github.com/state-spaces/mamba) library by Gu & Dao.

## Contact

For questions, issues, or collaboration: open a GitHub issue or contact `<your-email>`.