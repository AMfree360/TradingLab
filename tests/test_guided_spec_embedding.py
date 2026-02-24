import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.spec import StrategySpec
from gui_launcher import app as guided_app


def test_write_and_load_spec_with_x_guided_ui(tmp_path):
    # create minimal spec with long side so validation passes
    import uuid
    spec_name = 'TEST_GUIDED_SPEC_' + str(uuid.uuid4()).split('-')[0]
    spec_dict = {
        'name': spec_name,
        'description': 'test',
        'market': {'symbol': 'BTCUSDT', 'exchange': 'BINANCE', 'market_type': 'spot'},
        'entry_tf': '1h',
        'long': {'enabled': True, 'conditions_all': [{'tf': '1h', 'expr': 'true'}]},
    }
    spec = StrategySpec.model_validate(spec_dict)

    extra = {
        'context_rules': [{'type': 'price_vs_ma', 'length': 50, 'side': 'long'}],
        'signal_rules': [{'type': 'ma_cross', 'fast': 10, 'slow': 50, 'side': 'short'}],
        'trigger_rules': [],
    }

    # write to repo (research_specs)
    p = guided_app._write_validated_spec_to_repo(spec=spec, old_spec_name='', extra_meta=extra)
    assert p.exists()

    # load via compiler to ensure x_guided_ui is exposed
    from research.compiler import StrategyCompiler

    compiler = StrategyCompiler(repo_root=Path.cwd())
    loaded = compiler.load_spec(p)
    # compiler.load_spec attaches x_guided_ui attribute when present
    assert hasattr(loaded, 'x_guided_ui')
    assert isinstance(loaded.x_guided_ui, dict)
    assert 'context_rules' in loaded.x_guided_ui

    # cleanup
    p.unlink()
