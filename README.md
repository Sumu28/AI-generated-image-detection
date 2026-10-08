# Spotting AI-Generated Product Images

**Can a model tell a real product photo from one made by Midjourney, DALL·E or Stable Diffusion?**

Online marketplaces are filling up with product images that were never photographed. A buyer looking at a listing has little way of knowing whether the item in the picture exists. This project builds a classifier that looks at a product image and decides whether it is an **authentic photo (0)** or **AI-generated (1)**.

It started as the CSC1187 Machine Learning assignment at Dublin City University, set up as a competition on e-commerce product listings, and I treated it as a chance to compare two very different kinds of image model on a problem that matters.

---

## The two models

**Vision Transformer (ViT-Base-Patch16-224).** A transformer that splits an image into patches and uses self-attention to relate every patch to every other one. It sees the whole picture at once, which is useful when the giveaways of a fake are subtle and spread across the image. I fine-tuned the full model.

**EfficientNet-B4.** A convolutional network that scales depth, width and resolution together. It is a strong, well-proven baseline, and I wanted to know whether the newer transformer approach would actually beat it.

Both started from pretrained weights (transfer learning). ViT was fully fine-tuned, and EfficientNet was partially fine-tuned.

## How I approached it

1. **Preparing the data:** image preprocessing, normalisation, augmentation, and a train/validation split.
2. **Training experiments:** transfer learning, fine-tuning, learning rate scheduling and hyperparameter tuning, plus some experiments with ensembling the two models.
3. **Scoring:** the main metric was **F1**, because the classes were not evenly balanced and the competition used it too. Accuracy alone can look good while missing the images you most want to catch.

---

## Results

| Model | Validation F1 | Notes |
|---|---|---|
| ViT-Base/16-224 | **0.8721** | Best |
| EfficientNet-B4 | 0.7876 | Partial fine-tune |
| Ensemble (0.8 ViT / 0.2 EfficientNet) | 0.77 | Below ViT on its own |

These are scores on the validation set. *(If the competition had a leaderboard, add your test score and rank here, but only if you are sure of them.)*

**What I found**

- The Vision Transformer beat EfficientNet-B4 by about 8.5 F1 points (0.8721 vs. 0.7876). One caveat: EfficientNet was only partially fine-tuned, so a fully fine-tuned version might have narrowed the gap.
- **Ensembling did not help.** Blending the two models (80% ViT, 20% EfficientNet) scored 0.77, which is worse than ViT alone. The weaker model seems to have pulled the stronger one down rather than adding anything.

**Why I think ViT won.** AI-generated images often have small inconsistencies that show up across the whole image rather than in one spot, such as lighting that does not quite agree with itself or textures that repeat oddly. Self-attention looks at the image globally, so it may be better placed to notice these. This is my interpretation of the results, not something I have proved. I did not run tools like Grad-CAM to see what the models were actually looking at, which is the first thing on my list below.

## Limitations

- The models were trained on images from a fixed set of generators. New diffusion models produce different artefacts, so the detector may not carry over.
- I have not visualised what the models focus on, so the "global inconsistencies" explanation is a hypothesis.
- The scores above are validation scores from one dataset. *(Add anything else that applies, such as image resolution limits or the validation approach.)*

## What I would try next

- Grad-CAM, to check whether the model really is looking at the artefacts I think it is
- Larger Vision Transformers, Swin Transformers and ConvNeXt
- Revisiting ensembling, for example with two models of similar strength or a learned blend instead of fixed weights
- Fully fine-tuning EfficientNet-B4 for a fairer comparison
- Testing on images from newer diffusion models

---

## What is in this repository


```
source_code.ipynb                  # full training and evaluation notebook
Final_Esty_Assignment_Report.pdf   # the written report
submission_vit.csv                 # ViT predictions
submission_effnet.csv              # EfficientNet predictions
requirements.txt
README.md
```

## Tech stack

Python · PyTorch · Hugging Face Transformers · timm · NumPy · Pandas · Matplotlib · Scikit-learn

## Run it yourself

```bash
git clone https://github.com/Sumu28/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook source_code.ipynb
```

Then run the notebook from top to bottom to train and evaluate both models.

## Author

Sumukha Sagar
School of Computing, Dublin City University

## License

MIT
