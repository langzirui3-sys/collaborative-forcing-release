Rolling joint-attention diagnostic attribution

`src/cfexp/rolling_joint_handoff.py` adapts the cached inference branch of
`long_video/wan/modules/causal_model.py` from thu-ml/Causal-Forcing at commit
`eb8705b2f00b7c361193b25aa3aae9e7edc94667`, distributed under Apache-2.0
(see the adjacent LICENSE). Copyright 2024-2025 The Alibaba Wan Team Authors.

Changes: group accepted clean B3 queries and a first noisy B3 query in one
diagnostic model invocation; retain native projection geometries, per-group
attention extraction, the unrotated anchor, cursor writes and bounded-cache
evictions; add explicit admission checks and an optional clean-prefix witness.
No ordinary session or production factory selects this diagnostic.

`tests/fixtures/rolling_eb8705_native_attention.py.txt` contains the unmodified
`causal_rope_apply` and `CausalWanSelfAttention` definitions from that file,
and the unmodified `WanRMSNorm` definition from `long_video/wan/modules/model.py`
at the same commit. Imports are supplied by the differential test. The upstream
Git blob IDs reported by the repository file API are respectively
`5e7ca469b0176fefbecae3a119b56aa9eeca3054` and
`f8fa92742160d694fb81f572adf913e389f91b5a`.
