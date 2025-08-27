# Contributing Guidelines

## Notebook Output Stripping

We use `nbstripout` to automatically remove Jupyter notebook output before committing so that diffs stay readable and merge conflicts are minimized.

### One-Time Local Setup

Run:

```
poetry run nbstripout --install
```

This installs the Git filter in your local clone (it uses the `.gitattributes` entry `*.ipynb filter=nbstripout`).

### Manually Strip a Notebook

```
poetry run nbstripout path/to/notebook.ipynb
```

### Disable (Temporarily)

```
poetry run nbstripout --uninstall
```

(You can reinstall after with the first command.)

### Why

-   Keeps repository size smaller
-   Avoids noisy JSON diffs of cell outputs / execution counts
-   Reduces merge conflicts

## General Workflow

1. Create a feature branch
2. Add/modify code & notebooks (outputs will be stripped on commit)
3. Add concise, meaningful commit messages
4. Open a Pull Request

Thanks for contributing! 🎉
