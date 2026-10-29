# Flu-Prediction

Predicting next season's influenza hemagglutinin (HA) and neuraminidase (NA) gene sequences for the
H1N1 and H3N2 strains. The project compares the classical models from the original research
(Decision Tree, Random Forest, Extra Trees and Naive Bayes) with three deep learning models
(CNN, Transformer and BiLSTM).

**To understand what was done, start with [`relatorios/o que foi feito.pdf`](relatorios/)** (in Portuguese).
The full analysis of the results is in `relatorios/relatório e análise.pdf`.

## Installation

Requires Python 3.10 or newer. Inside the project folder:

```shell
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

# Only with an NVIDIA GPU (makes the neural networks much faster):
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Without an NVIDIA GPU, skip the `--index-url` line: `requirements.txt` installs the CPU build of PyTorch.
If the CUDA command fails, pick the right one at https://pytorch.org/get-started/locally/.

## Usage

```shell
python main.py --dataset all --align --seeds 5      # main experiment (about 1h40 on an MX570 GPU)
python main.py --dataset all --align                # single run, faster
python main.py --dataset h1n1-ha --models classical # only the original research's models
python main.py --models copy,random-forest,transformer --epochs 300 --verbose
```

| Option | What it does |
|---|---|
| `--dataset` | `h1n1-ha`, `h1n1-na`, `h3n2-ha`, `h3n2-na` or `all` |
| `--models` | groups (`baseline`, `classical`, `deep`, `all`) or comma-separated model names |
| `--align` | trims every sequence at its start codon (ATG) so positions line up |
| `--seeds` | repeats every model with several random seeds and saves the mean and standard deviation |
| `--epochs` | training epochs for the neural networks (default: 200) |
| `--device` | `auto`, `cuda` or `cpu` |

Results are printed and saved to `results/`:

- `<dataset>-aligned-seeds-summary.csv`: mean and standard deviation per model (**the final results**);
- `<dataset>-aligned-seeds.csv`: one row per model and seed;
- `<dataset>-aligned.csv` and `<dataset>.csv`: single runs, with and without alignment;
- `run-*.txt`: the full console output of the runs used in the reports.

## Models

| Name (`--models`) | Type | How it works |
|---|---|---|
| `copy` | baseline | repeats the previous season's sequence; any useful model must beat it |
| `decision-tree`, `random-forest`, `extra-trees`, `naive-bayes` | classical | the original research's models, on the A=1, T=2, G=3, C=4 encoding in 15-digit chunks |
| `cnn` | deep learning | convolutional network that reads the DNA one nucleotide at a time |
| `transformer` | deep learning | Transformer that reads the gene in codons (3-letter words) |
| `lstm` | deep learning | bidirectional LSTM over codons |

The neural networks include a copy mechanism: a learned shortcut that favours keeping the input
nucleotide, since almost all of the gene stays the same from one season to the next.

## Evaluation

Every model uses the original research's splits (2-fold cross-validation and a 70% train / 30% test
split with `random_state=50`) and gets the same scores:

- **nucleotide accuracy:** share of nucleotides predicted correctly;
- **mutation accuracy:** accuracy only on positions that changed between the old and new sequence
  (the `copy` model always scores 0% here);
- **R², MAE and MSE** on the 15-digit encoding (the original research's metrics).

## Project structure

```
main.py               entry point
dataset.py            FASTA loading, alignment and data preparation
evaluation.py         splits, baseline model and metrics
encoding_data.py      the original research's encoding
prediction_model/     classical models; neural networks in deep/
results/              CSV results and console output of the runs
relatorios/           PDF reports
flu-data/             FASTA sequences used in the experiments
```

| Report | Contents |
|---|---|
| `relatorios/o que foi feito.pdf` | summary of the work, the code and how to run it (Portuguese) |
| `relatorios/relatório e análise.pdf` | full analysis of the results with 5 seeds (Portuguese) |
| `relatorios/report and analysis.pdf` | first report, single run (English) |

## Notes

- **Bug in the original code:** the loop that read the target sequences used `data_test[i]` instead of
  `data_test[j]`, so every target was the same sequence and the results were inflated. It is fixed;
  the reports have the details.
- **GPU variation:** the neural networks can vary by about 0.05 points between runs with the same seed.
  The classical models always give the same result.
- **BiLSTM exit code:** on Windows, after the BiLSTM runs on the GPU, Python may exit with code
  -1073740791. This happens inside the NVIDIA driver after everything is saved and does not affect the results.

## Authors

- **[Marcos Paulo Gomes](https://github.com/Marcospgp26)**: this work (code update, bug fix, deep learning
  models and evaluation).
- **[Lucas Nascimento](https://www.linkedin.com/in/lucasnmacedo/)**: original code and data, from his
  undergraduate thesis in Computer Science at the Federal University of Uberlândia, inspired by Koodli's
  [RK900/Flu-Prediction](https://github.com/RK900/Flu-Prediction).

## License

MIT. See `LICENSE`.
