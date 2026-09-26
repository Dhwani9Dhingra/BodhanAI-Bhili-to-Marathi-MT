# Notebooks

The Colab notebook will be added after the core Python pipeline is
complete.

The notebook will NOT contain duplicated training logic.

It will only orchestrate repository commands such as:

```bash
python -m scripts.preflight
python -m scripts.prepare_data
python -m scripts.run_baselines
python -m scripts.train
python -m scripts.verify_adapter
python -m scripts.evaluate
python -m scripts.analyze