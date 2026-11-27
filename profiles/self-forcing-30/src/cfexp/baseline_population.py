"""Select the explicitly frozen four-pair table population without changing v1."""
import hashlib
import json
from pathlib import Path


def model_pair_for_draft(draft_model_id: str) -> str:
    pairs = {"self_forcing_1.3b": "krea_self_forcing", "longlive_1.3b": "krea_longlive", "rolling_forcing_1.3b": "krea_rolling_forcing"}
    if draft_model_id not in pairs:
        raise ValueError(f"unsupported population Draft identity: {draft_model_id}")
    return pairs[draft_model_id]


def load_baseline_population(root: Path, *, model_pair: str = "krea_self_forcing"):
    definition = root / 'experiments/main-table-population-v2.protocol.json'
    table = json.loads(definition.read_text())
    extension = None
    if model_pair == 'krea_rolling_forcing':
        extension_path = root / 'experiments/erank3-rolling-population-v1.protocol.json'
        extension = json.loads(extension_path.read_text())
        if (extension['schema'] != 'rolling_population_extension_v1'
                or extension['base_protocol'] != definition.name
                or extension['base_sha256'] != hashlib.sha256(definition.read_bytes()).hexdigest()
                or extension['model_pair'] != model_pair
                or extension['draft_model_id'] != 'rolling_forcing_1.3b'):
            raise ValueError('Rolling population extension binding mismatch')
    if extension is None and model_pair not in {item['id'] for item in table['model_pairs']}:
        raise ValueError('model pair is outside the frozen population protocol')
    parent = root / 'experiments' / table['prior_protocol']['file']
    if hashlib.sha256(parent.read_bytes()).hexdigest() != table['prior_protocol']['sha256']:
        raise ValueError('baseline population parent hash drifted')
    population = json.loads(parent.read_text())
    prompts = [{'id': p['prompt_id'], 'text': p['prompt']} for p in population['prompts']]
    if len(prompts) != 24 or len({p['id'] for p in prompts}) != 24 or population['seed'] != 20260911:
        raise ValueError('baseline population count, identities or seed drifted')
    binding = {'protocol': definition.name,
                     'sha256': hashlib.sha256(definition.read_bytes()).hexdigest(),
                     'model_pair': model_pair, 'seed': population['seed']}
    if extension is not None:
        binding['extension'] = {**extension, 'file': extension_path.name,
            'sha256': hashlib.sha256(extension_path.read_bytes()).hexdigest()}
    return prompts, binding


def select_prompt_shard(prompts, index, count):
    if type(index) is not int or type(count) is not int or not 1 <= count <= len(prompts) or not 0 <= index < count:
        raise ValueError('invalid prompt shard')
    return prompts[index::count]
