# Contributing to Celeste Client

Thanks for your interest in contributing to **Celeste Client**! We welcome issues and pull requests that improve the project.

## Getting Started

1. **Fork** the repository and create your feature branch from `main`.
2. Install dependencies with `uv pip install -e . --group dev`.
3. Install the pre-commit hooks with `pre-commit install`.
4. Create your changes on a new branch.

## Coding Standards

- Format and lint your code by running `pre-commit run --files <changed files>`.
- Keep functions small and well documented using docstrings.
- When adding new features, include tests where applicable.

## Making a Pull Request

1. Ensure your branch is rebased on the latest `main`.
2. Run `pre-commit run --files <changed files>` and fix any issues.
3. Push your branch and open a pull request describing your changes.

## Add a new Provider (concise)

1. Create client
- File: `src/celeste_client/providers/<provider>.py`
- Class: `<ProviderName>Client(BaseClient)` with:
  - `__init__`: `super().__init__(model=model, **kwargs)` + SDK client init
  - `async generate_content(prompt, **kwargs) -> AIResponse`
  - `async stream_generate_content(prompt, **kwargs) -> AsyncIterator[AIResponse]`

2. Wire factory
- Edit `src/celeste_client/__init__.py`:
  - Add to `SUPPORTED_PROVIDERS`
  - Add to `provider_mapping`: `Provider.MYPROVIDER: (".providers.myprovider", "MyProviderClient")`

3. Register a model
- Edit `celeste-core/src/celeste_core/models/catalog.py`:
  - Add a `Model(id=..., provider=Provider.<X>, capabilities=Capability.TEXT_GENERATION, ...)`

4. Dependencies/settings
- Add SDK to `pyproject.toml` if needed; ensure env keys in `celeste_core.config.settings`.

5. Consistency
- Use `BaseClient` for capability validation; pass-through `**kwargs`; stream yields `is_stream_chunk=True`.

6. Smoke test
- `uv run streamlit run example.py` and test both generate/stream.

We appreciate every contribution. Thank you for helping make Celeste Client better!
