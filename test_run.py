import sys
import traceback
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

try:
    import app
    print('Imported app OK')

    df = app.create_dummy_data(rows=50)
    print('create_dummy_data -> rows:', len(df))

    proc = app.process_data(df)
    print('process_data -> metrics keys:', list(proc['metrics'].keys()))
    print('process_data -> top_parameters count:', len(proc.get('top_parameters', [])))

    print('SMOKE TEST: SUCCESS')
except Exception as e:
    print('SMOKE TEST: FAILED')
    traceback.print_exc()
    raise
