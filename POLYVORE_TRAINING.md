# Polyvore Training

This project can use the `xthan/polyvore-dataset` metadata to train an optional
compatibility head for the API.

## 1. Download metadata

```bash
mkdir -p data/polyvore
curl -L https://github.com/xthan/polyvore-dataset/raw/master/polyvore.tar.gz \
  -o data/polyvore/polyvore.tar.gz
tar -xzf data/polyvore/polyvore.tar.gz -C data/polyvore
```

The GitHub archive contains the train/valid/test outfit metadata and
`fashion_compatibility_prediction.txt`. It does not contain the original item
images.

## 2. Add item images

Place the image dump under something like `data/polyvore/images`. The trainer
can resolve either of these common layouts:

```text
data/polyvore/images/119704139_1.jpg
data/polyvore/images/119704139/1.jpg
```

The original Polyvore URLs in the metadata are no longer available, so images
need to come from Kaggle, Google Drive, Hugging Face, or another mirror.

## 3. Check image coverage

```bash
venv/bin/python scripts/train_polyvore.py \
  --metadata-dir data/polyvore \
  --images-dir data/polyvore/images \
  --dry-run
```

## 4. Train the compatibility head

The default training mode follows the supplied compatibility guidance:
positive pairs come from items in the same curated outfit, and negative pairs
are sampled by replacing one item with a random item from the same category.
The default ratio is 3 negatives per positive.

```bash
venv/bin/python scripts/train_polyvore.py \
  --metadata-dir data/polyvore \
  --images-dir data/polyvore/images \
  --example-source category-negatives \
  --negative-ratio 3 \
  --epochs 5 \
  --batch-size 32
```

This writes `elephante_match_head.pt`. The FastAPI app auto-loads that file on
startup and uses it instead of plain cosine similarity.

To use the repository's prebuilt labeled compatibility file instead:

```bash
venv/bin/python scripts/train_polyvore.py \
  --metadata-dir data/polyvore \
  --images-dir data/polyvore/images \
  --example-source compatibility-file
```

## 5. Read the metrics

The trainer follows the evaluation guidance in
`Outfit Compatibility Model_ Data, Implementation, and Evaluation.pdf`:

```text
epoch=1 train_loss=... train_auc=... train_acc=... train_f1=... \
val_loss=... val_auc=... val_acc=... val_f1=... val_confusion=tp:.../fp:.../tn:.../fn:...
```

Validation AUC is the primary offline metric when both positive and negative
examples are present. Accuracy, precision, recall, F1, and the confusion counts
are logged to catch false positives and false negatives before using the model
in the API.

## 6. Metric hierarchy

Use this order while improving the model:

1. Compatibility AUC: main offline training metric for match/no-match labels.
2. FITB or FITB-hard accuracy: best next benchmark for retrieval-style outfit completion.
3. Stylist approval: human check for whether high-scoring matches are actually wearable.
4. Keep rate or purchase success: production metric once recommendations affect users.

Accuracy, precision, recall, F1, and confusion counts are supporting metrics;
they help debug, but they should not be the only target.

## 7. Failure-mode checklist

Review high-confidence false positives and false negatives for these patterns:

- Coat dominance: the model blames a small accessory when the coat changes the outfit context.
- Improper triangles: A matches B and B matches C, but A does not truly match C.
- Visual bias: backgrounds or photography style drive the score instead of garments.
- Scale/context mismatch: street photos, flat-lay shots, and oversized items behave differently.
- Same-category false positives: multiple tops look visually related but do not form an outfit.

## 8. Other training paths to consider

The current API path is a lightweight ResNet compatibility model. The supplied
research notes also point to larger or more specialized routes:

- LookBench notebooks for retrieval evaluation and custom model benchmarking.
- ResNet50 feature extraction notebooks for similarity-search baselines.
- Siamese-network notebooks for pair/triplet training.
- DeepLabV3+ segmentation notebooks when background or body-context bias becomes a blocker.
- Vision-language fine-tuning, such as Llama vision models, for richer styling explanations.

Those are useful next stages, but Polyvore compatibility AUC is the first
stable target for this server because it maps directly to `match_score`.
