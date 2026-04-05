# CBR-to-SQL: Rethinking Retrieval-based Text-to-SQL using Case-based Reasoning in the Healthcare Domain

## Installation and Setup

This project is configured with `uv`. To set it up, run:
```
uv init
uv sync
```

You will also need a Docker container for Qdrant to host the vector database. Make sure you have `docker-compose`, then run:
```
docker-compose up
```

## Quick Start

All runnable scripts live in the `scripts/` folder and use Interactive Python. Within `scripts/`, focus on the dataset-specific subfolders, which contain the scripts tailored to each dataset. Prompts, metrics, definitions, and model abstractions all live in `src/`. 

To ingest the training data into the knowledge base, run `build_knowledge_base.py`.

To build the lookup table, run `build_lookup_table.py`.

To evaluate performance, run `eval_performance.py`. Set the `EVALUATE` variable (e.g. `EVALUATE = ["CBR-CDB"]`) to select which models to evaluate. Additional configs for ablation studies can be found in `src/configs.py`. You may also need to change the `USE_AZURE` variable depending on your use case. 

## Acknowledgements

We reused many components from MIMICSQL's github repo for our use case: https://github.com/wangpinggl/TREQS. We appreciate the authors for their valuable code.