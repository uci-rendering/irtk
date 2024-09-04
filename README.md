# irtk
[![PyPI version](https://badge.fury.io/py/irtk.svg?kill_cache=1)](https://badge.fury.io/py/irtk)
[![documentation](https://github.com/uci-rendering/inv-render-toolkit/actions/workflows/documentation.yml/badge.svg)](https://uci-rendering.github.io/irtk/)

`irtk` stands for **I**nverse-**R**endering **T**ool**k**it. It is a simple framework for conducting inverse rendering experiments. Since there are a lot differentiable renderers in development, mastering them all and migrating between them can be a headache. `irtk` provides a simple scene representation based on `pytorch` that can be rendered by **connectors** that connect to different differentiable renderers and get the graident of the scene parameters. Migrating between renderers is just a matter of changing the connector. 

## Installation
`irtk` depends on `pytorch`. Since it is widely used and some projects might have specific requirements to it, we leave its installation to the user. 

After installing `pytorch`, you can install `irtk` via
```
pip install irtk
```
