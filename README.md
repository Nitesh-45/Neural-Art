# 🎨 Neural-Art

> AI-powered neural style transfer: apply the artistic style of one image to the content of another using PyTorch, AdaIN, and a Flask web app.

![Demo](Demo_IO_Images/NeuralArtvideo1.gif)
<!-- TODO: Add a before/after image or a short GIF. You already have a Demo_IO_Images folder, so reuse one example from there. -->

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C)
![Flask](https://img.shields.io/badge/Flask-webapp-000000)
![License](https://img.shields.io/badge/License-MIT-green)

**🔗 Live Demo:** `https://neural-art-64zs.onrender.com` &nbsp;

---

## 📌 Problem
Style transfer lets anyone turn a normal photo into artwork. Early methods were slow because they optimised each image separately. This project uses **Adaptive Instance Normalization (AdaIN)**, which stylises any content/style pair in a single forward pass.

## ✨ Features
- **Two-image style transfer:** keep the structure of a content image, take the look of a style image
- **AdaIN-based transformation** of channel-wise feature statistics
- **Adjustable style strength** using an alpha value between 0 and 1
- **VGG-based encoder** and a **trainable convolutional decoder**
- **Flask web interface** for uploading images and viewing results
- Automatic **GPU/CPU selection** (CUDA if available)
- Decoder **training script** with content and style losses

## 🛠️ Tech Stack
| Area | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Model | VGG encoder + decoder, AdaIN |
| Image processing | Pillow |
| Web app | Flask, Flask-WTF, WTForms |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Server config | Gunicorn |

## 🏗️ How It Works
```
Content Image ──> VGG Encoder ──> Content Features ──┐
                                                     ├──> AdaIN ──> Alpha Blend
Style Image ─────> VGG Encoder ──> Style Features ───┘                  │
                                                                        ▼
                                                               Trained Decoder
                                                                        │
                                                                        ▼
                                                               Stylized Image
```

**AdaIN** moves the channel-wise mean and standard deviation of the style features onto the content features:

```
AdaIN(x, y) = σ(y) · ((x − μ(x)) / σ(x)) + μ(y)
```

The final features are blended with alpha: `t = α · AdaIN(x, y) + (1 − α) · x`

- `alpha = 0` → original content preserved
- `alpha = 1` → full style applied

**Training loss:** `L = λc · L_content + λs · L_style`, where content loss is the MSE between output features and the AdaIN target, and style loss is the MSE between channel means and standard deviations across encoder levels. Optimiser: Adam with a LambdaLR inverse-decay schedule.

## 📊 Results

| Setting | Value |
|---|---|
| Training epochs | 200 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Content weight / Style weight | 1.0 / 10 |
| Training datasets | Content: COCO test2017 (40k images) <br> Style: WikiArt (8k images) |
| Inference time (CPU / GPU) | ~200 ms (CPU) / ~20 ms (GPU) for 256x256 image |

![Before and after](Demo_IO_Images/i-p/i_p%20image.jpg)
![Before and after](Demo_IO_Images/o-p/o_p%20style%201.jpg)
<!-- TODO: Replace with a real example from Demo_IO_Images -->

## 🚀 Installation

```bash
git clone https://github.com/Nitesh-45/Neural-Art.git
cd Neural-Art

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
```

Install the PyTorch build for your system from the [official selector](https://pytorch.org/get-started/locally/), then:
```bash
pip install -r requirements.txt
```

### Model checkpoints
Place these files before running (they are not stored in the repo):
```
NST_Code/
├── vgg_normalised.pth
└── experiment/
    └── final_exp/
        └── decoder_final.pth
```
`https://github.com/Nitesh-45/Neural-Art/releases/tag/v1.0.0`

## ▶️ Usage

**Run the web app**
```bash
cd NST_Code
python app.py
```
Open `http://localhost:5000`, upload a content image and a style image, set alpha, and generate. Supported formats: `.png`, `.jpg`, `.jpeg`.

**Train the decoder**
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

## 📁 Project Structure
```
├── Demo_IO_Images/         # Example inputs and outputs
├── NST_Code/
│   ├── app.py              # Flask app and inference pipeline
│   ├── train.py            # Decoder training script
│   ├── utils/
│   │   ├── models.py       # VGG encoder and decoder
│   │   └── utils.py        # Dataset, transforms, AdaIN, statistics
│   ├── templates/index.html
│   └── experiment/final_exp/options.txt
├── code.ipynb              # Exploratory VGG feature-analysis notebook
├── requirements.txt
└── README.md
```

## ⚠️ Limitations & Future Work
- Checkpoints must be supplied separately, and `app.py` still has a machine-specific decoder path that should be made relative
- Set `SECRET_KEY` from an environment variable, validate alpha in `[0, 1]`, limit upload size and use unique filenames
- Rename `Procfile.txt` to `Procfile` for platforms like Heroku/Render
- Planned: unit tests for AdaIN, downloadable outputs, before/after comparison view, reproducible training config

## 🙏 Credits
- Huang & Belongie, [*Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868)


## 👤 Author
**Nitesh Singh Mehra** · [LinkedIn](https://www.linkedin.com/in/nitesh-singh-mehra-18nsm) · [GitHub](https://github.com/Nitesh-45)
