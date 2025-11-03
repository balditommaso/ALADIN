import os
import re
import subprocess
import pytest

from network_generate import network_generate

network_pairs = [
    (
        {
            'frontend': 'QONNX',
            'target': 'PULP.PULP_gvsoc',
            'conf_file': "./models/checkpoint/config_files/config_QONNX_dummy_mix_bits.json",
            'optional': 'mixed-sw'
        },
        {
            'frontend': 'QONNX',
            'target': 'PULP.PULP_gvsoc',
            'conf_file': "./models/checkpoint/config_files/config_QONNX_dummy_mix_bits_with_lut.json",
            'optional': 'mixed-sw'
        }
    ),
]



# --- utility: run one network and get checksum + raw stdout
def run_network(network_args, appdir='./application'):
    network_generate(**network_args)

    cmd = ['make', '-C', appdir, 'clean', 'all', 'run', 'platform=gvsoc', 'CORE=8']
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=360)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Build or run failed (exit {e.returncode}):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"Test timed out after 360s.\nPartial STDOUT:\n{e.output}\nSTDERR:\n{e.stderr}")

    output = proc.stdout
    print(output)  # captured unless run with -s

    match = re.search(r"Final output[^\n]*:\s*\n?([-\d\s]+)", output)
    if not match:
        pytest.fail(f"No logits found in output:\n{output[:500]}")

    logits = list(map(int, re.findall(r"-?\d+", match.group(1))))
    return logits



@pytest.mark.parametrize('net1_args, net2_args', network_pairs)
def test_compare_networks(net1_args, net2_args, capsys):
    appdir = './application'

    with capsys.disabled():
        print(f"Comparing networks:\n  A: {net1_args['conf_file']}\n  B: {net2_args['conf_file']}")
        
        logits1 = run_network(net1_args, appdir)
        logits2 = run_network(net2_args, appdir)
        # Run both networks
    
    with capsys.disabled():
        print(f"Logits {net1_args['conf_file']}:\n {logits1}")
        print(f"Logits {net2_args['conf_file']}:\n {logits2}")

    # Compare
    assert logits1 == logits2
