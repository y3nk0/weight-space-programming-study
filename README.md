# Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models: the notebook

Konstantinos Skianis and Aristidis Likas, Department of Computer Science and Engineering, University of Ioannina.

This repository accompanies the paper of the same title. It holds one notebook, [`edit_demo.ipynb`](edit_demo.ipynb), together with the data it needs. The notebook explains the setting on a small model and re-checks results of the paper on seven of its runs.

## The setting in a few lines

A frontier language model (the editor) is given the weights of a smaller language model, a Python session, sixteen examples of a task and a budget of fifty forward passes. It is asked to edit the weights, without gradients, so that the small model solves the task with no examples in the prompt. The paper compares what the editors write with in-context learning (ICL) and with fine-tuning on the same sixteen examples.

## What the notebook does

It needs no API key and it runs no editor. Every weight change it applies was written during the experiments of the paper and is stored in `runs/`. The notebook loads Qwen2.5-0.5B from Hugging Face, applies a stored change, measures the edited model and prints each number next to the one stored for the paper. A line that ends in `ok` means the two agree.

1. **The model.** What Qwen2.5-0.5B contains, and which numbers an edit touches.
2. **In-context learning and the task vector.** The model fails without examples and succeeds with sixteen. The difference between the two is a vector, and adding it by hand is enough.
3. **One edit, checked closely.** The weight change that Gemini 4 Argon wrote for the antonym task: 13 hidden units, one per layer. The notebook shows that the units fire on the prompt format and stay silent on ordinary text, that what they write is the shift that the sixteen examples cause, and that ordinary language modelling is untouched.
4. **Six more edits.** The same measurements on runs of Claude Fable 5.1, Claude Opus 5.5 and GPT-6 Astra on the other five tasks.

## The seven runs

All are on Qwen2.5-0.5B. Accuracy is exact match on the test items of the task; perplexity is measured on WikiText-2 and is 17.34 for the untouched model.

| run | editor | task | numbers changed | zero-shot accuracy before | after the edit | 16-shot ICL | perplexity after |
|---|---|---|---|---|---|---|---|
| `antonym_gemini150_s0` | Gemini 4 Argon | word to its opposite | 34,944 | 0.0% | 90.5% | 93.7% | 17.35 |
| `capital_fable_s2` | Claude Fable 5.1 | country to capital | 2,688 | 2.6% | 90.4% | 96.5% | 17.37 |
| `past_tense_fable_p50` | Claude Fable 5.1 | verb to past tense | 8,959 | 0.0% | 92.7% | 93.5% | 17.44 |
| `past_tense_astra_s0` | GPT-6 Astra | verb to past tense | 4,358,140 | 0.0% | 92.7% | 93.5% | 17.34 |
| `en_fr_opus_s0` | Claude Opus 5.5 | English to French | 1,788 | 0.0% | 83.7% | 86.0% | 17.34 |
| `times_2_opus_s0` | Claude Opus 5.5 | n to 2n | 43,008 | 0.0% | 96.5% | 76.9% | 17.34 |
| `add_7_astra_s0` | GPT-6 Astra | n to n+7 | 29,567 | 0.0% | 100.0% | 78.5% | 18.12 |

The model has 494,032,768 parameters. The numbers in the table are the ones stored for the paper; the notebook recomputes them, and a value can differ in its last digit from one machine to another. These seven runs are a sample: the paper reports all runs, including those that did not finish.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab edit_demo.ipynb
```

- A GPU is not required. The notebook uses one if it finds one and the CPU otherwise. On a laptop CPU the whole notebook takes about seven minutes.
- The first run downloads the model (about 1 GB) and the WikiText-2 test text from Hugging Face.
- The notebook opens with its outputs saved, so it can be read without running it.
- The same code runs without Jupyter: `.venv/bin/python edit_demo.py` prints the same checks.
- `torch` needs a Python that matches the machine. On an Apple-silicon Mac this means an arm64 Python, for example `/usr/bin/python3`.

## What is in the repository

```
edit_demo.ipynb          the notebook, with its outputs
edit_demo.py             the same notebook as a plain Python script
runs/<run>/delta.pt      the weight change the editor left
runs/<run>/run.json      the editor's final explanation, its turns, forward passes and cost
runs/<run>/task_prompt.md  the prompt the editor received
results/v2/<run>.json    the evaluation of the run made for the paper
results/unit_report.json per-unit measurements for the runs that write key-value units
wsp/common.py            the prompt format and the exact-match scorer
wsp/tasks.py             the task data and the 16 / 30 / rest split
figures/                 the drawing of the model shown in the notebook
```

The harness that runs the editors, the API keys it needs, the transcripts and the remaining runs are not part of this repository.

## Citation

```bibtex
@article{skianis2026weightspace,
  title   = {Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models},
  author  = {Skianis, Konstantinos and Likas, Aristidis},
  year    = {2026},
  note    = {Preprint}
}
```

## Use of AI assistance

Claude (Anthropic) was used as a writing and presentation assistant for this project: drafting and editing prose, and producing the tables, figures, slides and results page from the experimental outputs. The research questions, experimental design, analysis, interpretation and conclusions are the authors' own.

## License

Code is released under the MIT License. The editors' explanations in `runs/*/run.json` belong to the transcripts of the study, which are released under CC BY 4.0.
