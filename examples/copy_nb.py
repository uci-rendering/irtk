from pathlib import Path
import shutil

# copy all notebooks to docs/source/examples

doc_path = Path('..', 'docs', 'source', 'examples')

for p in Path('.').glob('*.ipynb'):
    shutil.copy(p, doc_path)
