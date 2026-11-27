"""Reserve a native two-GPU slot while checking its primary GPU.

This lets the slot queue put a one-GPU parity gate before a two-GPU cost
probe without racing a second job into the unused device. It is not a
multi-GPU performance measurement. Formal runner leases stay unchanged.
"""
import argparse
import os
import subprocess
import sys
from .main_table_baseline_campaign import _visible_devices
from .main_table_ours_campaign import _assert_idle_pair


def run(model):
    modules={'longlive':'cfexp.longlive_chunk_parity','rolling':'cfexp.rolling_forcing_parity'}
    if model not in modules:raise ValueError('unsupported native parity model')
    devices=_visible_devices()
    if len(devices)!=2 or len(set(devices))!=2:raise ValueError('two distinct reserved GPUs required')
    _assert_idle_pair(*devices)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[0]))
    subprocess.run([sys.executable,'-m',modules[model]],env=env,check=True)
    _assert_idle_pair(*devices)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=('longlive','rolling'))
    run(p.parse_args().model)

if __name__=='__main__':main()
