# SLM - Small Language Model

This codebase contains code for pretraining and posttraining a Language Model for the software project [Building Language Models from Scratch](https://ypdu.github.io/courses/building-language-models-from-scratch/) instructed by Dr. Yupei Du.

## Installation

To install locally, use 

```bash
uv sync
```
This will install the the package along with the dev dependencies. The repo needs packages like `jaxtyping`, `beartype` and `pytest` to run static and runtime typechecking and coverage of tests.

### Development & Testing

A good way to contribute to this repo is by following the TDD (**Test Driven Development**) philosophy- for each feature that you would like to add, plan and write a comprehensive test for its behaviour, output (shapes, etc). Keep in mind that each component must be well documented (look at PEP257). An important component of documentation is to add typehints (PEP 484). We go a step further by using `jaxtyping` to also mention shapes of tensors that are received as input / returned by functions (look at attention in `src/slm/functional.py`). The repo uses the Google styleguide / docstring format. A very easy way to easily get a template for each function / class is to use the vscode extension `autoDocstring` by Nils Werner. 

To run tests (and do a health check of the repo at a particular point), run 

```bash
uv run pytest
```



## Preparing Dataset

```bash

```

## Launching Pretraining

```bash

```

## Evaluations

```bash

```

## Posttraining

```bash

```
