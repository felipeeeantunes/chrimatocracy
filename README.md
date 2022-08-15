# DataScienceBR
Data Science applied to the government sector

## Build
You'll need to install [Poetry](https://python-poetry.org/) and set a list of environment variables:

### Poetry Installation
We are using the `1.1.10` version, type:
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.1.10
```
Add `export PATH="/Users/<user.name>/.local/bin:$PATH"` to your shell configuration file (eg. ~/.zshrc) and restart your terminal and check if Poetry was installed correctly.

```bash
poetry --version
```

## Straightforward instructions for setting up and running Chirmatocracy

These are provisory instructions for working with provisory branches.


1. Clone the project:
    ```bash
    git clone https://github.com/felipeeeantunes/chrimatocracy.git
    ```

1. Switch to the specific branch you are going to work on (`feature/structuring-modules`, for instance):
    ```bash
    git switch feature/structuring-modules
    ```

Go inside the directory and build the environment:
    ```bash
    cd chrimatocracy
    make build
    ```
    Ps. You can use the Poetry commands to create local environments and builds instead.

5. Code patterns
   ```bash
   # run these commands below before adding files to commit
   make isort
   make black
   # then, git add <files>...
   ```