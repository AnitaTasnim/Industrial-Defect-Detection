
# 🧠 PatchCore with EfficientNet-B5 on MVTec AD

## 📌 Overview

This project performs **defect detection (anomaly detection)** on the **MVTec AD dataset** using the **PatchCore** model from the [Anomalib](https://github.com/openvinotoolkit/anomalib) library.
It uses a **custom EfficientNet-B5 feature extractor** to replace the original `TimmFeatureExtractor` which causes issues in `Anomalib 0.7.0`.

---

## ⚙️ Key Features

* ✅ Custom **EfficientNet-B5** feature extractor
* ⚡ PatchCore anomaly detection model
* 📦 Automatic dataset copy from `/kaggle/input`
* 🚀 GPU-accelerated training with PyTorch Lightning
* 📊 Anomaly heatmaps and ROC-AUC evaluation
* 🧩 Fully reproducible configuration (YAML)

---

## 📁 Project Structure

```
PatchCore_with_EfficientNetB5_on_MVTecAD/
│
├── PatchCore_with_EfficientNetB5_on_MVTecAD.ipynb   # Main notebook
├── README.md                                        # Documentation
│
├── mvtec-ad/                                        # Dataset (copied from Kaggle input)
│   ├── bottle/
│   ├── cable/
│   └── ...
│
├── models/                                          # (Optional) Saved model weights
├── outputs/                                         # Anomaly maps, visual results
└── configs/
    └── patchcore_config.yaml                        # Modified Anomalib config
```

---

## 🧩 Custom Feature Extractor

```python
class EfficientNetFeatureExtractorWrapper(nn.Module):
    """
    Custom wrapper for EfficientNet feature extraction.
    Replaces TimmFeatureExtractor in Anomalib 0.7.0.
    """
    def __init__(self, backbone_name: str, layers: list[str]):
        super().__init__()
        base_model = timm.create_model(backbone_name, pretrained=True)
        self.layer_indices = [int(layer.split('.')[-1]) for layer in layers]
        self.conv_stem = base_model.conv_stem
        self.bn1 = base_model.bn1
        self.blocks = base_model.blocks

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        x = self.conv_stem(inputs)
        x = self.bn1(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.layer_indices:
                outputs[f"blocks.{i}"] = x
        return outputs
```

---

## 🧪 Dataset

**Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
**Categories:** 15 industrial objects (e.g., bottle, cable, capsule…)
**Data Types:** Normal vs. Defective samples

Dataset auto-copied from Kaggle input:

```python
ORIGINAL_DATASET_PATH = "/kaggle/input/mvtec-ad"
OUTPUT_DIR = "/kaggle/working/"
DATASET_PATH = os.path.join(OUTPUT_DIR, "mvtec-ad")

if not os.path.exists(DATASET_PATH):
    print(f"Copying dataset from {ORIGINAL_DATASET_PATH} to {DATASET_PATH}...")
    shutil.copytree(ORIGINAL_DATASET_PATH, DATASET_PATH)
else:
    print("Dataset already exists.")
```

---

## 🚀 Training

Train PatchCore on all MVTec AD categories:

```python
trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)
trainer.fit(model=model, datamodule=datamodule)
```

Train per category:

```python
for category in all_categories:
    print(f"Training category: {category}")
    ...
```

---

## 📊 Evaluation

After training, the model produces:

* Pixel-wise anomaly maps
* Image-level anomaly scores
* ROC-AUC metrics

Visualize example results:

```python
plt.imshow(anomaly_map, cmap="inferno")
plt.title("Anomaly Heatmap")
plt.show()
```

---

## 💡 Example Results

| Category | Image-AUC | Pixel-AUC |
| -------- | --------- | --------- |
| Bottle   | 0.991     | 0.983     |
| Cable    | 0.978     | 0.965     |
| Capsule  | 0.984     | 0.971     |

*(Values vary depending on configuration and epochs.)*

---

## 🧰 Dependencies

```bash
pip install anomalib==0.7.0 timm pytorch-lightning omegaconf opencv-python matplotlib torch torchvision
```

---

## ⚙️ Configuration (YAML)

```yaml
model:
  name: patchcore
  backbone: efficientnet_b5
  layers: ["blocks.2", "blocks.5", "blocks.8"]
  input_size: 224
  embedding_size: 550
  anomaly_map_mode: pixel
```

---

## 📈 Future Improvements

* [ ] Integrate ViT and ConvNeXt backbones
* [ ] Add mixed precision (AMP) for faster training
* [ ] Build anomaly dashboard visualization

---

## 👩‍💻 Author

**Anita Tasnim**
Machine Learning & Computer Vision Engineer
📧 [Your Email Here]
🔗 [Your Kaggle or GitHub Link Here]

 
