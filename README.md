# Neural-Art
### AI-Powered Neural Style Transfer with Adaptive Instance Normalization

Transform the visual content of one image using the artistic style of another. Neural-Art combines a VGG-based feature encoder, Adaptive Instance Normalization (AdaIN), and a trainable decoder in a Flask web application.

</div>

---

## Overview

Neural-Art is an image-to-image style transfer application built with **PyTorch** and **Flask**. It takes two inputs:

- **Content image:** the photograph or scene whose structure should be preserved.
- **Style image:** the artwork or image whose visual appearance should be transferred.

The model extracts deep feature representations from both images, aligns the content features with the style statistics using AdaIN, and reconstructs the stylized output through a decoder.

The web interface supports image uploads, previews, a style-strength control, and generated output display.

## Key Features

- **Two-image style transfer:** combine the structure of a content image with the appearance of a reference style image.
- **AdaIN-based transformation:** adapt channel-wise feature statistics of the content representation to those of the style representation.
- **Adjustable style strength:** control the interpolation between the original content features and stylized features using the alpha parameter.
- **VGG feature extraction:** use a VGG-based encoder to capture hierarchical visual representations.
- **Flask web interface:** upload images and view the generated result through a browser.
- **GPU/CPU device selection:** the inference and training code select CUDA when available and otherwise use the CPU.
- **Custom decoder training:** train the decoder with content and style losses.

## How It Works

```text
Content Image ──> VGG Encoder ──> Content Features ──┐
                                                     ├──> AdaIN ──> Alpha Blend
Style Image ─────> VGG Encoder ──> Style Features ───┘                  │
                                                                        v
                                                               Trained Decoder
                                                                        │
                                                                        v
                                                               Stylized Image
```

### Adaptive Instance Normalization

AdaIN transfers the channel-wise mean and standard deviation of the style features to the content features:

\[
\operatorname{AdaIN}(x,y)=
\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\mu(y)
\]

Where:

- \(x\) represents content features.
- \(y\) represents style features.
- \(\mu\) is the channel-wise mean.
- \(\sigma\) is the channel-wise standard deviation.

The final feature representation is blended using:

\[
t_{\alpha}=\alpha\,\operatorname{AdaIN}(x,y)+(1-\alpha)x
\]

- `alpha = 0`: preserve the original content features.
- `alpha = 1`: apply the full AdaIN transformation.
- Values between `0` and `1`: balance content preservation and style strength.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Vision models | VGG-based encoder and convolutional decoder |
| Style transfer method | Adaptive Instance Normalization (AdaIN) |
| Image processing | Pillow, Torchvision |
| Web framework | Flask |
| Forms and validation | Flask-WTF, WTForms |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Production server configuration | Gunicorn |

## Project Structure

```text
ai-nst-project/
├── Demo_IO_Images/              # Input/output demonstration images
├── NST_Code/
│   ├── app.py                   # Flask application and inference pipeline
│   ├── train.py                 # Decoder training script
│   ├── utils/
│   │   ├── models.py            # VGG encoder and decoder architectures
│   │   └── utils.py             # Dataset, transforms, AdaIN and statistics
│   ├── templates/
│   │   └── index.html           # Web application interface
│   └── experiment/
│       └── final_exp/
│           └── options.txt      # Recorded experiment configuration
├── code.ipynb                   # Exploratory VGG feature-analysis notebook
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nitesh-45/ai-nst-project.git
cd ai-nst-project
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the PyTorch build appropriate for your system from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

## Model Checkpoints

The application requires pretrained model-weight files that are not included in the visible repository file listing:

- `vgg_normalised.pth` — weights loaded by the VGG encoder.
- `decoder_final.pth` — trained decoder checkpoint used for inference.

Place the files at the following locations, or update the paths in `NST_Code/app.py` to match your local setup:

```text
NST_Code/
├── vgg_normalised.pth
└── experiment/
    └── final_exp/
        └── decoder_final.pth
```

**Important:** The current `app.py` contains a machine-specific absolute path for `decoder_final.pth`. Replace it with a project-relative path before running the application. For portability, resolve model paths relative to `app.py` and load checkpoints with an explicit `map_location=device`.

Example path setup:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VGG_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"
```

Then initialize the models using `VGG_PATH` and `DECODER_PATH`.

## Run the Web Application

After installing dependencies and placing the required checkpoints:

```bash
cd NST_Code
python app.py
```

Open the local address printed by Flask. The current application is configured to run on:

```text
http://localhost:5000
```

Upload a content image and a style image, adjust the alpha value, and submit the form to generate the stylized image.

Supported upload extensions in the current application: `.png`, `.jpg`, and `.jpeg`.

## Train the Decoder

The training script accepts content-image and style-image directories, VGG weights, experiment settings, and optimization parameters.

Example:

```bash
cd NST_Code

python train.py \
  --content_dir ./content_data \
  --style_dir ./style_data \
  --vgg ./vgg_normalised.pth \
  --experiment Neural-Art_run \
  --batch_size 16 \
  --epochs 200 \
  --lr 0.0001 \
  --content_weight 1.0 \
  --style_weight 10
```

The training script writes experiment outputs under:

```text
NST_Code/experiment/<experiment-name>/
```

This includes recorded arguments and, when the save interval is reached, decoder checkpoints, optimizer checkpoints, and sample output images.

The checked-in experiment configuration records a 200-epoch run with batch size 16, learning rate `0.0001`, content weight `1.0`, and style weight `10`. The training script's own defaults differ, so pass the intended values explicitly when reproducing that configuration.

## Training Objective

The decoder is optimized using a weighted combination of content and style losses:

\[
L_{\text{total}}= \lambda_c L_{\text{content}}+\lambda_s L_{\text{style}}
\]

- **Content loss:** mean-squared error between the generated image's encoded features and the AdaIN-transformed target features.
- **Style loss:** mean-squared error between the channel-wise means and standard deviations of generated-image features and style-image features across encoder levels.
- **Optimizer:** Adam.
- **Learning-rate scheduling:** LambdaLR with inverse decay.

## Exploratory Notebook

`code.ipynb` contains an exploratory analysis intended to visualize VGG feature activations for original and stylized images. It is separate from the Flask inference flow and should be treated as an analysis notebook rather than a required application component.

## Current Implementation Notes

- Model checkpoints must be supplied separately; the visible repository listing does not show the required `.pth` weight files.
- Update the hard-coded decoder checkpoint path in `app.py` before deployment.
- Set a secure Flask `SECRET_KEY` through an environment variable instead of keeping a fixed secret in source code.
- Validate the alpha value to keep it within the interval `[0, 1]`.
- Use unique upload filenames and restrict upload size; the current implementation saves uploads using their sanitized original filenames.
- The deployment file is named `Procfile.txt`; platforms expecting a file named `Procfile` may not detect it. The Gunicorn target must also be run from a directory where the `app` module is importable.
- The notebook contains incomplete or malformed cells in its current form and may need corrections before it runs end-to-end.

## Potential Improvements

1. Make all model, dataset, upload, and output paths configurable.
2. Add upload-size limits, robust image validation, unique filenames, and cleanup of temporary files.
3. Add unit tests for AdaIN, feature extraction, alpha blending, and image preprocessing.
4. Add inference timing and memory monitoring for CPU and GPU execution.
5. Add downloadable output images and a before/after comparison view.
6. Add a reproducible training configuration file and document the source and licensing of datasets and checkpoints.

## References

- [AdaIN: Arbitrary Style Transfer in Real-Time](https://arxiv.org/abs/1703.06868)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

<div align="center">

**Neural-Art** · Neural Style Transfer with PyTorch and AdaIN

</div>


