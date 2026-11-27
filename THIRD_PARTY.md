# Third-party attribution

The Rolling Forcing profile includes two files marked Apache-2.0 that
adapt the Causal Forcing implementation.

| File in the Rolling Forcing source profile | Adaptation |
| --- | --- |
| `src/cfexp/rolling_forcing_stream.py` | Incremental finalized blocks, configuration checks, RNG isolation and cache release |
| `src/cfexp/rolling_joint_handoff.py` | Rolling clean-history updates and the first resumed Draft window using native anchor handling |

The Rolling Forcing test fixture `tests/fixtures/rolling_eb8705_native_attention.py.txt`
contains reference definitions from that upstream source and retains the
Alibaba Wan Team copyright, source revision, and Apache-2.0 identifier.

Keep the source headers and the original license and attribution in
`docs/licenses/causal-forcing/` with the adapted files. The upstream revision
named in that attribution is `eb8705b2f00b7c361193b25aa3aae9e7edc94667`.
[Causal Forcing source](https://github.com/thu-ml/Causal-Forcing/tree/eb8705b2f00b7c361193b25aa3aae9e7edc94667) ·
[Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0)
