## How to build the documentation
The documentation will be automatically built and reflect the updates on GitHub page upon push. For development, follow the instructions below.

Install the following via `pip`:

```bash
pip install sphinx sphinx-autobuild furo
```

At the root directory of this repo, run
```bash
sphinx-autobuild docs/source docs/build/html
```