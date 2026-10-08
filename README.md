# Spotting AI-Generated Product Images

**Can a model tell a real product photo from one made by Midjourney, DALL·E or Stable Diffusion?**

Online marketplaces are filling up with product images that were never photographed. A buyer looking at a listing has little way of knowing whether the item in the picture exists. This project builds a classifier that looks at a product image and decides whether it is an **authentic photo (0)** or **AI-generated (1)**.

It started as the CSC1187 Machine Learning assignment at Dublin City University, set up as a competition on e-commerce product listings, and I treated it as a chance to compare two very different kinds of image model on a problem that matters.

---

## The two models

**Vision Transformer (ViT-Base-Patch16-224).** A transformer that splits an image into patches and uses self-attention to relate every patch to every other one. It sees the whole picture at once, which is useful when the giveaways of a fake are subtle and spread across the image. I fine-tuned the full model.

**EfficientNet-B4.** A convolutional network that scales depth, width and resolution together. It is a strong, well-proven baseline, and I wanted to know whether the newer transformer approach would actually beat it.

Both started from pretrained weights (transfer learning) and were fine-tuned on the task.

## How I approached it

1. **Preparing the data:** image preprocessing, normalisation, augmentation, and a train/validation split.
2. **Training experiments:** transfer learning, fine-tuning, learning rate scheduling and hyperparameter tuning, plus some experiments with ensembling the two models.
3. **Scoring:** the main metric was **F1**, because the classes were not evenly balanced and the competition used it too. Accuracy alone can look good while missing the images you most want to catch.

*(Add the dataset here: where it came from, how many images, and the real/AI split. For example, "X training images, Y% AI-generated".)*

---

## Results

| Model | F1 score |
|---|---|
| Vision Transformer (ViT) | **[add your F1]** |
| EfficientNet-B4 | [add your F1] |

*(If the competition had a leaderboard, add your score and rank, but only if you are sure of them.)*

**What I found**

- The Vision Transformer beat EfficientNet-B4.
- Fine-tuning made a big difference compared with using the pretrained models as they were. *(Add the before and after numbers.)*

**Why I think ViT won.** AI-generated images often have small inconsistencies that show up across the whole image rather than in one spot, such as lighting that does not quite agree with itself or textures that repeat oddly. Self-attention looks at the image globally, so it may be better placed to notice these. This is my interpretation of the results, not something I have proved. I did not run tools like Grad-CAM to see what the models were actually looking at, which is the first thing on my list below.

## Limitations

- The models were trained on images from a fixed set of generators. New diffusion models produce different artefacts, so the detector may not carry over.
- I have not visualised what the models focus on, so the "global inconsistencies" explanation is a hypothesis.
- Results come from one dataset. *(Add anything else that applies, such as image resolution limits or the validation approach.)*

## What I would try next

- Grad-CAM, to check whether the model really is looking at the artefacts I think it is
- Larger Vision Transformers, Swin Transformers and ConvNeXt
- A proper ensemble of the best models
- Testing on images from newer diffusion models

---

## What is in this repository

*(Check this against the real files, and consider renaming the report to fix the "Esty" typo.)*

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
