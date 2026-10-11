# CD Extraction from SEM Images using U-Net

Deep-learning pipeline that segments synthetic SEM (Scanning Electron Microscope) images with a U-Net in PyTorch and extracts Critical Dimensions (CDs) from the predicted masks.

In semiconductor manufacturing, CDs (feature widths, pitch, height) are measured from SEM images after patterning steps. Doing this by hand is slow and subject to human bias. This project shows an automated alternative: segment the image into material regions, then measure CDs geometrically from the segmentation.

## Highlights

- Synthetic SEM-style dataset of **300 images** with **5 material classes**
- Randomized structural parameters, realistic corner rounding and SEM-noise simulation
- Lightweight **U-Net** (`base_filters=16`) trained in PyTorch (Apple MPS device)
- CD extraction by geometric scanning of predicted masks
- **Sub-2% mean error** on gate width, pitch and height on the synthetic test set

## Pipeline

```
Notebook 1: synthetic data generation
        |
        v
data/ (images, integer label masks, ground_truth.csv)
        |
        v
Notebook 2: U-Net training  ->  models/unet_sem_cd.pth
        |
        v
Predicted masks -> geometric scan -> CD values -> compare to ground truth
```

## Repository structure

```
.
├── 01_generate_dataset.ipynb   # synthetic SEM dataset generation
├── 02_train_and_extract_cd.ipynb  # U-Net training + CD extraction + evaluation
├── data/                       # generated images, masks, ground_truth.csv (not tracked)
├── models/                     # saved weights, e.g. unet_sem_cd.pth (not tracked)
├── requirements.txt
└── README.md
```

Rename the notebook filenames above to match your actual files.

## Method

**1. Dataset generation**
Notebook 1 draws randomized structure parameters (gate width, pitch, height, etc.), renders the layered structure, applies corner rounding, and adds SEM-like noise. Each image has an integer label mask for its five material classes, and true CD values are saved to `ground_truth.csv`.

**2. Segmentation**
A lightweight U-Net (encoder-decoder with skip connections, `base_filters=16`) is trained with a 70/15/15 train/validation/test split for 15 epochs. Skip connections preserve the fine spatial detail needed for accurate edge locations.

**3. CD extraction**
The predicted mask is scanned geometrically (row/column crossings between classes) to measure gate width, pitch and height in pixels, which can be converted to nm using the image scale.

## Results

| Metric | Mean error (synthetic test set) |
|---|---|
| Gate width | < 2% |
| Pitch | < 2% |
| Height | < 2% |

Results are on synthetic data. Real SEM images have more noise, charging artifacts and contrast variation, so expect lower accuracy without fine-tuning.

## Getting started

```bash
git clone https://github.com/<maity-rupam>/<sem_cd_extraction_unet>.git
cd <sem_cd_extraction>
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `01_generate_dataset.ipynb` to create `data/`
2. `02_train_and_extract_cd.ipynb` to train the model and evaluate CD extraction

Requirements: Python 3.9+, PyTorch, NumPy, pandas, matplotlib, Jupyter. The code uses the MPS device on Apple Silicon and can fall back to CUDA or CPU.

## Limitations and future work

- Trained and tested only on synthetic images
- Fixed set of five material classes and simple structure types
- Next steps: fine-tune on real SEM images, compare with classical methods such as level-set thresholding, add uncertainty and flagging for ambiguous measurements, and try a transformer-based segmentation model

## Related work

- Medina et al., "Automated extraction of critical dimension from SEM images with Weave", Proc. SPIE 11611 (2021): level-set segmentation approach
- Samir et al., "CD-SEM Image Defect Detection and Classification Using Transformers", J. Electrical Systems 20-10s (2024): DETR for defect detection

## Author

Rupam Maity, MTech, Centre for Nano Science and Engineering (CeNSE), IISc
