---
title: GridSync — Power Plant Net Output Predictor
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
license: mit
tags:
  - pytorch
  - regression
  - energy
  - gradio
  - ann
  - power-plant
  - deep-learning
  - ccpp
---

<div align="center">

# ⚡ GridSync
### Combined Cycle Power Plant — Net Output Predictor

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Spaces-Deployed-FFD21E)](https://huggingface.co/spaces/KARTHIKAKRISHNA123/GridSync)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade ANN regression pipeline that predicts net electrical energy output (PE) of a combined cycle power plant from 4 ambient environmental parameters — deployed on Hugging Face Spaces with zero infrastructure overhead.**

</div>

---

## 📌 Problem Statement

Combined Cycle Power Plants (CCPP) integrate gas turbines, steam turbines, and heat recovery to maximize energy conversion efficiency. Net electrical output (PE) is highly sensitive to ambient conditions — temperature, vacuum pressure, atmospheric pressure, and humidity — all of which vary hour by hour.

**This system answers:** *Given current environmental sensor readings, what net power output (MW) will the plant generate?*

Accurate prediction enables grid operators to balance load, plan maintenance windows, and optimize dispatch — reducing both waste and cost.

---

## 🎯 Solution Overview

GridSync wraps a trained PyTorch regression ANN inside a Gradio Blocks interface. Users set 4 environmental sliders (bounded by real CCPP dataset min/max values) and receive an instant power output prediction with a visual load bar and operating-level classification.

The model was trained with best-checkpoint saving — only the epoch with the lowest validation MSE is persisted as `gridsync_model.pth`.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Regression output** | Continuous MW prediction (operating range: 420 – 496 MW) |
| **4 input parameters** | AT (temperature), V (vacuum), AP (pressure), RH (humidity) |
| **Visual load bar** | `█░░` unicode bar showing % of rated operating range |
| **Load classification** | High / Medium / Low load level label |
| **Data-grounded sliders** | Min/max/default derived from 9,568-sample CCPP dataset |
| **Best-model checkpoint** | `best_model.pt` saved at lowest validation MSE epoch |
| **Industrial dark-teal UI** | Grid-monitoring aesthetic; 4-card stat header |

---

## 🏗️ Overall Architecture

```mermaid
flowchart LR
    subgraph Input["Input Layer"]
        U["👤 Operator\n(4 Sliders)"]
    end

    subgraph Preprocessing["Preprocessing"]
        SC["StandardScaler\nscaler.pkl"]
    end

    subgraph Model["ANN Regression Model\ngridsync_model.pth"]
        L1["Linear 4→6\n+ ReLU"]
        L2["Linear 6→6\n+ ReLU"]
        L3["Linear 6→1\n(no activation)"]
        L1 --> L2 --> L3
    end

    subgraph Postprocessing["Post-processing"]
        CL["Clamp to\n420–496 MW"]
        PCT["% of range\n+ load label"]
    end

    subgraph Output["Output"]
        R["MW prediction\n+ load bar\n+ table"]
    end

    U -->|"AT, V, AP, RH"| SC
    SC -->|"scaled tensor [1×4]"| L1
    L3 -->|"raw float (MW)"| CL
    CL --> PCT --> R
```

---

## 🧠 System Architecture

```mermaid
flowchart TD
    subgraph HFSpace["Hugging Face Space (Gradio SDK)"]
        APP["app.py\n(Entrypoint)"]

        subgraph Artifacts["Serialized Artifacts"]
            PTH["gridsync_model.pth\nBest-epoch weights"]
            SCLR["scaler.pkl\nStandardScaler"]
        end

        subgraph GradioBlocks["Gradio Blocks UI"]
            HDR["HTML Header"]
            STATS["Stat Grid (4 cards)"]
            AT_S["AT Slider\n1.81 – 37.11 °C"]
            V_S["V Slider\n25.36 – 81.56 cm Hg"]
            AP_S["AP Slider\n992.89 – 1033.30 mbar"]
            RH_S["RH Slider\n25.56 – 100.16 %"]
            BTN["Predict Button"]
            OUT["gr.Markdown Output"]
        end

        subgraph Inference["Inference Pipeline"]
            NP["np.array (1×4)"]
            TF["scaler.transform()"]
            TEN["torch.tensor float32"]
            FWD["model.forward()"]
            CLAMP["max-min clamp"]
            FMT["Markdown formatter"]
        end
    end

    APP --> Artifacts
    APP --> GradioBlocks
    BTN -->|"on.click()"| NP
    NP --> TF --> TEN --> FWD --> CLAMP --> FMT --> OUT
```

---

## 🧰 Technology Stack — Complete Breakdown

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| **PyTorch** | 2.x | Deep Learning | ANN regression model definition, training, and inference | Dynamic graph, `state_dict` portability, best-checkpoint save pattern | `nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `nn.MSELoss`, `optim.Adam`, `model.eval()`, `torch.no_grad()`, `torch.save()`, `torch.load()` |
| **torch.nn** | — | Model API | Layered MLP architecture for regression | Sequential API enables clean serialization and loading | `nn.Sequential`, `nn.Linear(4,6)`, `nn.Linear(6,6)`, `nn.Linear(6,1)`, `nn.ReLU` |
| **torch.optim** | — | Optimization | Adam optimizer for weight updates | Adaptive learning rate; converges faster than SGD on tabular data | `optim.Adam(model.parameters())` |
| **scikit-learn** | 1.x | Preprocessing | Feature standardization of AT, V, AP, RH | Zero-mean unit-variance improves gradient flow in early layers | `StandardScaler.fit_transform(X_train)`, `StandardScaler.transform(X_test)` |
| **joblib** | — | Serialization | Persist fitted `scaler.pkl` for inference reuse | Efficient numpy-array serialization, preferred over pickle for sklearn | `joblib.dump()`, `joblib.load()` |
| **NumPy** | 1.x | Numerical | Construct input array from slider values | Bridge between Python float → sklearn scaler → PyTorch tensor | `np.array([[at,v,ap,rh]], dtype=np.float32)` |
| **pandas** | 2.x | Data I/O | Load and split `powerplant_data.csv` during training | DataFrame-native train/test split with `.values` to numpy | `pd.read_csv()`, `df.drop()`, `y.values` |
| **Gradio** | 4.x | UI / Serving | Blocks layout with sliders and markdown output | Native HF Spaces SDK; `gr.Markdown` for rich formatted predictions | `gr.Blocks`, `gr.Slider`, `gr.Markdown`, `gr.Button`, `gr.Row`, `gr.HTML`, `gr.themes.Base`, CSS |
| **Gradio Themes** | — | Design | Industrial dark-teal colour tokens | Energy domain requires precision and authority — cold teal achieves this | `gr.themes.colors.teal`, `gr.themes.colors.cyan`, `gr.themes.GoogleFont` |
| **Matplotlib** | 3.x | Visualization | Training/validation loss curve during notebook analysis | Quick plot of train vs val MSE over 100 epochs | `plt.plot()`, `plt.legend()` — training only, not in inference |

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| **Source** | UCI ML Repository — Combined Cycle Power Plant |
| **File** | `powerplant_data.csv` |
| **Rows** | 9,568 hourly samples |
| **Features** | AT, V, AP, RH (4 environmental sensors) |
| **Target** | PE — Net Electrical Energy Output (MW) |
| **Train / Test split** | 80 / 20, `random_state=42` |

### Feature Statistics

| Feature | Description | Min | Mean | Max |
|---|---|---|---|---|
| **AT** | Ambient Temperature (°C) | 1.81 | 19.65 | 37.11 |
| **V** | Exhaust Vacuum (cm Hg) | 25.36 | 54.31 | 81.56 |
| **AP** | Ambient Pressure (mbar) | 992.89 | 1013.26 | 1033.30 |
| **RH** | Relative Humidity (%) | 25.56 | 73.31 | 100.16 |
| **PE** | Net Power Output (MW) | 420.26 | 454.37 | 495.76 |

---

## 🔄 Request Lifecycle

### Prediction Request — User Sets Environmental Conditions

```
1. USER INTERACTION
   └── User adjusts 4 Gradio sliders (AT, V, AP, RH)
       → Clicks "⚡ Predict Net Power Output"
       → btn.click(fn=predict, inputs=[at, v, ap, rh], outputs=output)

2. PYTHON FUNCTION CALL
   └── predict(at, v, ap, rh) invoked with 4 float arguments

3. PREPROCESSING
   └── np.array([[at, v, ap, rh]], dtype=np.float32)
       → shape: (1, 4)
       → scaler.transform(arr)   [StandardScaler from scaler.pkl]
       → output: (1, 4) zero-mean unit-variance

4. TENSOR CONVERSION
   └── torch.tensor(arr_scaled, dtype=torch.float32)
       → shape: [1, 4]

5. FORWARD PASS
   └── model.eval() + torch.no_grad()
       → Linear(4→6) + ReLU
       → Linear(6→6) + ReLU
       → Linear(6→1)           [raw float — no activation on output]
       → model(tensor).item()  [Python float]

6. POSTPROCESSING
   └── clamped  = max(420.0, min(496.0, output))
       → pct    = (clamped - 420.0) / 76.0 * 100
       → bar_str = "█" × int(pct/5) + "░" × remaining
       → load_label = High / Medium / Low

7. OUTPUT RENDER
   └── Markdown string with:
       - ## {MW} headline
       - Unicode bar + percentage
       - Table: raw prediction / range / load level
       → gr.Markdown renders in HF Space
```

---

## 🌊 Data Flow Explanation

```
Training Phase (Notebook)              Inference Phase (HF Space)
────────────────────────────           ─────────────────────────────
powerplant_data.csv                    Gradio sliders (AT, V, AP, RH)
         │                                          │
  pd.read_csv()                         np.array([[at,v,ap,rh]])
         │                                          │
  X = [AT,V,AP,RH], y = [PE]           shape: (1, 4) float32
         │                                          │
  train_test_split (80/20)              scaler.transform()     ← scaler.pkl
         │                                          │
  StandardScaler.fit_transform(X_train) scaled tensor [1, 4]
         │                                          │
  TensorDataset + DataLoader(batch=32)  model.forward()        ← gridsync_model.pth
         │                                          │
  ANN.forward() + MSELoss               raw float (MW)
         │                                          │
  Adam.step() × 100 epochs              clamp(420, 496)
         │                                          │
  if val_loss < best:                    pct-of-range + bar_str
    torch.save(state_dict)               │
         │                               gr.Markdown output
  joblib.dump(scaler)
```

---

<details>
<summary>📐 UML Diagram Suite — All 9 Diagrams</summary>

### 1. Use Case Diagram

```mermaid
graph TD
    OP(["👤 Grid Operator"])
    SYS(["⚡ GridSync Space"])

    UC1["Set Ambient Temperature"]
    UC2["Set Exhaust Vacuum"]
    UC3["Set Ambient Pressure"]
    UC4["Set Relative Humidity"]
    UC5["Submit Prediction Request"]
    UC6["View Power Output Estimate"]
    UC7["Interpret Load Level"]

    OP --> UC1
    OP --> UC2
    OP --> UC3
    OP --> UC4
    OP --> UC5
    UC5 --> UC6
    UC6 --> UC7
    SYS --> UC5
    SYS --> UC6
```

### 2. Class Diagram

```mermaid
classDiagram
    class ANN {
        +model: nn.Sequential
        +__init__()
        +forward(x: Tensor) Tensor
    }

    class StandardScaler {
        +mean_: ndarray
        +scale_: ndarray
        +fit_transform(X_train) ndarray
        +transform(X) ndarray
    }

    class InferencePipeline {
        +model: ANN
        +scaler: StandardScaler
        +predict(at, v, ap, rh) str
        -_clamp(val, lo, hi) float
        -_format_markdown(mw, pct) str
    }

    class GradioUI {
        +at_slider: gr.Slider
        +v_slider: gr.Slider
        +ap_slider: gr.Slider
        +rh_slider: gr.Slider
        +btn: gr.Button
        +output: gr.Markdown
        +launch()
    }

    InferencePipeline --> ANN
    InferencePipeline --> StandardScaler
    GradioUI --> InferencePipeline
```

### 3. Sequence Diagram

```mermaid
sequenceDiagram
    participant OP as Operator
    participant G as Gradio UI
    participant P as predict()
    participant SC as StandardScaler
    participant M as ANN Model
    participant FMT as Formatter
    participant OUT as gr.Markdown

    OP->>G: Set AT, V, AP, RH sliders
    OP->>G: Click Predict
    G->>P: predict(at, v, ap, rh)
    P->>P: np.array shape (1,4)
    P->>SC: transform(arr)
    SC-->>P: scaled_arr (1,4)
    P->>P: torch.tensor float32
    P->>M: model.forward(tensor)
    M-->>P: raw float MW
    P->>FMT: clamp + bar + table
    FMT-->>G: markdown string
    G-->>OUT: render
    OUT-->>OP: MW + load bar + table
```

### 4. Activity Diagram

```mermaid
flowchart TD
    A([Start]) --> B[User opens HF Space]
    B --> C[Stat cards load: 9568 samples, 420-496 MW range]
    C --> D[User adjusts 4 environment sliders]
    D --> E[Click Predict Net Power Output]
    E --> F[predict() invoked]
    F --> G[np.array constructed 1x4]
    G --> H[StandardScaler transforms]
    H --> I[FloatTensor created]
    I --> J[ANN forward pass]
    J --> K[Raw float extracted .item()]
    K --> L[Clamp to 420-496 MW]
    L --> M[Compute pct of range]
    M --> N[Build unicode bar and load label]
    N --> O[Format markdown string]
    O --> P[gr.Markdown renders]
    P --> Q([Operator reads MW estimate])
```

### 5. Component Diagram

```mermaid
flowchart LR
    subgraph UI2["Gradio Blocks UI"]
        CMP1B["Header — HTML"]
        CMP2B["Stat Grid — 4 cards"]
        CMP3B["AT Slider"]
        CMP4B["V Slider"]
        CMP5B["AP Slider"]
        CMP6B["RH Slider"]
        CMP7B["Predict Button"]
        CMP8B["Markdown Output"]
    end

    subgraph ART2["Artifacts"]
        CMP9B["gridsync_model.pth"]
        CMP10B["scaler.pkl"]
    end

    subgraph PIPE2["Inference"]
        CMP11B["predict()"]
    end

    CMP7B -->|"click"| CMP11B
    CMP11B --> CMP10B
    CMP11B --> CMP9B
    CMP11B --> CMP8B
```

### 6. Deployment Diagram

```mermaid
flowchart TD
    subgraph HF2["Hugging Face Infrastructure"]
        subgraph SPC2["GridSync Space - CPU Runtime"]
            APP3["app.py"]
            PTH3["gridsync_model.pth"]
            SCLR3["scaler.pkl"]
            REQ3["requirements.txt"]
        end
        BLD2["HF Build System\npip install + launch"]
    end

    subgraph DEV2["Developer Machine"]
        NB2["ANN_Regression.ipynb"]
        GP2["git push"]
    end

    NB2 -->|"torch.save + joblib.dump"| SPC2
    GP2 -->|"CD trigger"| BLD2
    BLD2 --> SPC2
    U3["👤 Browser"] -->|"HTTPS"| APP3
```

### 7. State Diagram

```mermaid
stateDiagram-v2
    [*] --> Idle: Space starts
    Idle --> InputReady: User adjusts sliders
    InputReady --> InputReady: More slider changes
    InputReady --> Computing: Click Predict
    Computing --> Rendering: predict() returns string
    Rendering --> Idle: gr.Markdown updated
    Computing --> Error: Exception in pipeline
    Error --> Idle: Gradio error toast shown
```

### 8. Object Diagram

```mermaid
flowchart LR
    OBJ1B["model: ANN\n──────────\ntraining = False\nbest-epoch weights loaded"]
    OBJ2B["scaler: StandardScaler\n──────────\nmean_ = [μAT, μV, μAP, μRH]\nscale_ = [σAT, σV, σAP, σRH]"]
    OBJ3B["tensor: FloatTensor\n──────────\nshape = [1, 4]\ndtype = float32"]
    OBJ4B["output: float\n──────────\nraw MW prediction\nclamped to 420-496"]

    OBJ3B --> OBJ1B
    OBJ2B -->|"scales"| OBJ3B
    OBJ1B -->|"forward pass"| OBJ4B
```

### 9. Package Diagram

```mermaid
flowchart TD
    PKG1B["app.py"]
    PKG2B["torch + torch.nn + torch.optim"]
    PKG3B["numpy"]
    PKG4B["joblib"]
    PKG5B["scikit-learn"]
    PKG6B["gradio"]
    PKG7B["pandas"]

    PKG1B --> PKG2B
    PKG1B --> PKG3B
    PKG1B --> PKG4B
    PKG1B --> PKG5B
    PKG1B --> PKG6B
    PKG7B -->|"training only"| PKG2B
```

</details>

---

<details>
<summary>📊 Data Flow Diagrams — L0 and L1</summary>

### DFD Level 0 — Context Diagram

```mermaid
flowchart LR
    E1C["👤 Grid Operator"]
    P0C(("0.0\nGridSync\nPower Prediction\nSystem"))
    E2C["📊 CCPP\nDataset"]

    E1C -->|"AT, V, AP, RH readings"| P0C
    P0C -->|"PE prediction in MW + load level"| E1C
    E2C -->|"9568 hourly training samples"| P0C
```

### DFD Level 1 — System Processes

```mermaid
flowchart TD
    E1D["👤 Grid Operator"]
    E2D["📊 CCPP Dataset"]

    P1D(("1.0\nReceive\nEnvironmental Input"))
    P2D(("2.0\nStandardize\nFeatures"))
    P3D(("3.0\nRun ANN\nForward Pass"))
    P4D(("4.0\nClamp and\nScale Output"))
    P5D(("5.0\nFormat and\nRender Result"))

    D1D[("D1: scaler.pkl\nStandardScaler")]
    D2D[("D2: gridsync_model.pth\nANN Weights")]

    E1D -->|"AT, V, AP, RH floats"| P1D
    E2D -->|"fit StandardScaler on X_train"| D1D
    E2D -->|"train ANN for 100 epochs"| D2D
    P1D -->|"raw ndarray 1x4"| P2D
    D1D -->|"mean and scale parameters"| P2D
    P2D -->|"scaled tensor 1x4"| P3D
    D2D -->|"layer weights and biases"| P3D
    P3D -->|"raw float MW value"| P4D
    P4D -->|"clamped MW and load pct"| P5D
    P5D -->|"markdown with bar and table"| E1D
```

</details>

---

## 📁 Folder Structure

```
GridSync/                            ← HF Space root (cloned repo)
├── app.py                           ← Entrypoint: ANN class + predict() + Gradio UI
├── requirements.txt                 ← Runtime dependencies
├── gridsync_model.pth               ← Best-checkpoint ANN weights (lowest val MSE)
├── scaler.pkl                       ← Fitted StandardScaler for AT, V, AP, RH
└── README.md                        ← This file (HF Space card + documentation)
```

---

## ⚙️ Prerequisites

- Python 3.10+
- Git with [Git LFS](https://git-lfs.github.com/) (for `.pth` files > 10MB)
- Hugging Face account + `huggingface_hub` CLI

---

## 🚀 Local Installation

```bash
# 1. Clone the Space
git clone https://huggingface.co/spaces/KARTHIKAKRISHNA123/GridSync
cd GridSync

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm artifacts exist
ls gridsync_model.pth scaler.pkl

# 4. Launch
python app.py
# → Opens at http://localhost:7860
```

---

## 🏋️ Training Artifacts — How to Export from Notebook

The notebook already saves `best_model.pt` during training. After training completes, rename and export:

```python
import joblib

# best_model.pt is already saved by the training loop
# Rename to match app.py expectation
import os
os.rename("best_model.pt", "gridsync_model.pth")

# Export scaler
joblib.dump(scaler, "scaler.pkl")

print("✅ Artifacts ready: gridsync_model.pth, scaler.pkl")
```

Move both files into the cloned HF Space directory before `git push`.

---

## 🧪 Inference Pipeline Internals

```python
# What predict() does step by step
arr        = np.array([[at, v, ap, rh]], dtype=np.float32)  # (1, 4)
arr_scaled = scaler.transform(arr)                            # (1, 4) standardized
tensor     = torch.tensor(arr_scaled, dtype=torch.float32)   # FloatTensor [1, 4]

with torch.no_grad():
    output = model(tensor).item()     # raw float — no activation on output layer

clamped = max(420.0, min(496.0, output))          # safety clamp to dataset range
pct     = (clamped - 420.0) / (496.0 - 420.0) * 100
```

---

## 📦 Dependencies

```
torch          — ANN architecture, weight loading, tensor ops
numpy          — input array construction
scikit-learn   — StandardScaler for feature normalization
joblib         — scaler serialization and loading
gradio         — Blocks UI, sliders, markdown output, HF Spaces serving
pandas         — CSV loading during training (not required at inference)
```

---

## 🚀 Deployment

```bash
cd GridSync/

git add app.py requirements.txt gridsync_model.pth scaler.pkl README.md
git commit -m "feat: initialize GridSync inference engine"
git push
# HF dashboard: Building → Running
```

> **Git LFS** — if `gridsync_model.pth` exceeds 10MB:
> ```bash
> git lfs install
> git lfs track "*.pth" "*.pkl"
> git add .gitattributes
> ```

---

## 🔒 Security Considerations

- All inputs are bounded sliders derived from dataset min/max — no string injection surface
- Model runs on CPU; no sensitive data persisted between requests
- Artifacts are read-only at inference time

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Inference latency | < 20ms on CPU |
| Model parameters | ~115 (4×6 + 6×6 + 6×1 + biases) |
| Memory footprint | < 100KB |
| Training samples | 9,568 |
| Best-epoch checkpoint | Saved at lowest validation MSE |

---

## 👩‍💻 Author

**Karthika Krishna M**  
B.E. Computer Science & Engineering  
Anna University Regional Campus, Tirunelveli  
Co-founder, Niranthara · AI/ML Engineer  

[![GitHub](https://img.shields.io/badge/GitHub-KARTHIKAKRISHNA123-181717?logo=github)](https://github.com/KARTHIKAKRISHNA123)
[![HuggingFace](https://img.shields.io/badge/🤗-KARTHIKAKRISHNA123-FFD21E)](https://huggingface.co/KARTHIKAKRISHNA123)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.