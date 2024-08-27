## How to build the documentation
The documentation will be automatically built and reflect the updates on GitHub page upon push. For development, follow the instructions below. Some dependencies requires `Python >= 3.10` for their latest version. Also, `irtk` need to be installed for automatically generating API references.p

Change to the `docs` directory:
```bash
cd docs
```

Install the following via `pip`:

```bash
pip install -r requirements.txt
```

Finally, run
```bash
sphinx-autobuild source build/html
```