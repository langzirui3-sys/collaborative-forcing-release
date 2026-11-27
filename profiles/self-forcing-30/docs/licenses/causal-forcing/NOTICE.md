Rolling Forcing inference attribution

`src/cfexp/rolling_forcing_stream.py` adapts the rolling inference algorithm in
`pipeline/rolling_forcing_inference.py` from thu-ml/Causal-Forcing, commit
`eb8705b2f00b7c361193b25aa3aae9e7edc94667`, distributed under Apache-2.0
(see the adjacent LICENSE).

Changes: exposes finalized blocks incrementally, validates the supported
configuration, isolates RNG state across pauses, and releases owned caches.
The initial-latent branch and VAE decoding are not included. This is not yet
a validated Target-intervention backend.
