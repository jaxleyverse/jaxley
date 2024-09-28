# Documentation

The documentation is available at: <https://jaxleyverse.github.io/jaxley/>

## Building the Documentation

You can build the docs locally by running the following command from this subfolder:

```bash
jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir mkdocs/tutorial/
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir mkdocs/examples/
mkdocs serve
```

The docs can be updated to GitHub using:

```bash
jupyter nbconvert --to markdown ../tutorials/*.ipynb --output-dir mkdocs/tutorial/
jupyter nbconvert --to markdown ../examples/*.ipynb --output-dir mkdocs/examples/
mkdocs gh-deploy
```
