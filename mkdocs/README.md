# Documentation

This folder used to host the _outdated_ mkdocs documentation, which was available at:
<https://jaxleyverse.github.io/jaxley/>.

This folder now only exists to maintain a minimal `mkdocs` website. You should not
be changing anything here, unless you are confident in what you are doing.

## Building the Documentation

You can build the docs locally by running the following command from this subfolder:

```bash
jupyter nbconvert --to markdown ../docs/tutorials/*.ipynb --output-dir docs/tutorial/
mkdocs serve
```

The docs can be updated to GitHub using:

```bash
jupyter nbconvert --to markdown ../docs/tutorials/*.ipynb --output-dir docs/tutorial/
mkdocs gh-deploy
```
