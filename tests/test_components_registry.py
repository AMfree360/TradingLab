import subprocess
import sys
import os

def test_components_register_fields():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    node = os.environ.get('NODE', 'node')
    script = os.path.join(repo_root, 'scripts', 'js_tests', 'test_components.js')
    res = subprocess.run([node, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = res.stdout.decode('utf-8', errors='replace')
    err = res.stderr.decode('utf-8', errors='replace')
    if res.returncode != 0:
        print('STDOUT:\n', out)
        print('STDERR:\n', err)
    assert res.returncode == 0, f'Node harness failed: {err}'