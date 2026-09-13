# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this branch is

Branch `dl6` holds EECS 498/598 (Michigan, Justin Johnson) **Assignment 6**: VAE, GAN,
network visualization, and style transfer. The repo (`main`) is a broader CV self-study
curriculum; each `dl*` branch is one assignment checked out at the repo root. The user is a
self-studier: no grades, no submission zip, no autograder.

**Read `.claude/CLAUDE.MD` first.** It is the user's session brief: working rules
(one function at a time, 🟢 = Claude writes it, 🔴 = user writes it and Claude only reviews,
be concise) and a progress table to update before ending a session.

## Layout

- Student files with `TODO` blocks (the only files to edit): `vae.py`, `gan.py`,
  `network_visualization.py`, `style_transfer.py`. Every block is delimited by a
  `# Replace "pass" statement with your code` / `END OF YOUR CODE` banner; `style_transfer.py`
  has inconsistent indentation inside `guided_gram_matrix` (2 spaces) — keep it valid.
- One driver notebook per file: `variational_autoencoders.ipynb`, `generative_adversarial_networks.ipynb`,
  `network_visualization.ipynb`, `style_transfer.ipynb`. Notebooks import the `.py` files with
  `%autoreload 2`, so edits to the modules take effect without restarting the kernel.
- `a6_helper.py`: provided plumbing (SqueezeNet preprocess/deprocess, `train_vae`, `show_images`,
  `initialize_weights`, `one_hot`, data loaders). It imports `loss_function` from `vae.py` at
  module top, so `vae.py` must at least import cleanly.
- `eecs598/`: course library. `eecs598.grad.rel_error` is the check used by every sanity cell;
  `eecs598.utils.reset_seed` is called before seeded checks. `eecs598/submit.py` is unused here.
- `images/`: content/style images plus `*_sky` / `*_nosky` masks for the spatial (guided)
  style transfer section.

## Running and verifying

There is no test suite. Verification is the notebooks' sanity cells, which compare against
hard-coded expected values with `rel_error` (target ~1e-7 or smaller; the notebooks print
the error rather than asserting):

- VAE: `reparametrize` mean/std check, `loss_function` expected `8.5079`, then 10-epoch
  MNIST training for `VAE` and `CVAE`.
- GAN: `sample_noise` shape/range check, `discriminator_loss`/`generator_loss` expected
  values (`1.8424` / `0.7713`), LS-GAN losses (`1.8770` / `0.8170`), exact parameter counts
  (`discriminator()` 267009, `generator(4)` 1858320, `build_dc_classifier()` 1102721,
  `build_dc_generator(4)` 6580801), then `run_a_gan` for FC, LS, and DC variants.
- Style transfer: `style-transfer-checks.npz` (downloaded with `style_data.zip`) gives
  expected outputs for `content_loss`, `gram_matrix`, `style_loss`, `tv_loss`.
- Network visualization has no numeric checks; results are judged visually
  (saliency maps, adversarial attack succeeds, class visualization).

Quick local import/syntax check without a notebook (`style_transfer.py` does
`from a6_helper import *`, so it needs scipy; the other three import with torch alone):

```bash
python3 -c "import vae, gan, network_visualization"
python3 -m py_compile style_transfer.py a6_helper.py
```

## Environment traps

- Notebooks were written for **Colab**: they hard-code `device='cuda'` / `.cuda()` and
  contain a Google Drive mount + `sys.path.append(GOOGLE_DRIVE_PATH)` section. Skip that
  section locally; the repo root is already the working directory.
- Local machine: torch 2.8 (CUDA build) is installed but **CUDA is not available**, and
  **scipy is not installed**, so `import a6_helper` fails locally. The style-transfer
  notebook calls `check_scipy()`. Treat the notebooks as things to run on Colab or a GPU box
  unless the user says otherwise.
- Data is downloaded lazily by the notebooks: MNIST to `./MNIST_data`,
  `imagenet_val_25.npz` to `./datasets/`, `style_data.zip` unpacked to `./styles/` plus
  `style-transfer-checks.npz`. None of it is in git.
- Code dates from 2022 and has a few stale idioms that break on current Python/libraries:
  `loader_train.__iter__().next()` in the VAE notebook (use `next(iter(loader_train))`),
  `torchvision.models.squeezenet1_1(pretrained=True)` (deprecated kwarg, still works with a
  warning), and `from scipy.ndimage.filters import gaussian_filter1d` in `a6_helper.py`
  (deprecated namespace; `scipy.ndimage` is the current path).
