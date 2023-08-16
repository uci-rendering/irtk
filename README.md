# irtk
`irtk` stands for Inverse-Rendering Toolkit. It is a simple framework for conducting inverse rendering experiments. Since there are a lot differentiable renderers in development, mastering them all and migrating between them can be a headache. `irtk` provides a simple scene representation based on `pytorch` that can be rendered by **connectors** that connect to different differentiable renderers and get the graident of the scene parameters. Migrating between renderers is just a matter of changing the connector. 

## Installation
`irtk` depends on `pytorch`. Since it is widely used and some projects might have specific requirements to it, we leave its installation to the user. 

After installing `pytorch`, you can install `irtk` via
```
pip install irtk
```