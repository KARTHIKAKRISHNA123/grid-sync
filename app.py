"""
GridSync — Power Plant Net Output Predictor
Architecture: Linear(4→6) → ReLU → Linear(6→6) → ReLU → Linear(6→1)
Dataset: Combined Cycle Power Plant (CCPP)
Features: AT, V, AP, RH  →  Target: PE (Net Electrical Energy Output, MW)
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import gradio as gr

# ── 1. Architecture (mirrors notebook exactly) ────────────────────────────────
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)

# ── 2. Load weights + scaler ──────────────────────────────────────────────────
model = ANN()
model.load_state_dict(torch.load("gridsync_model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")   # StandardScaler fitted on X_train (AT, V, AP, RH)

# ── 3. Inference ──────────────────────────────────────────────────────────────
def predict(at, v, ap, rh):
    arr        = np.array([[at, v, ap, rh]], dtype=np.float32)
    arr_scaled = scaler.transform(arr)
    tensor     = torch.tensor(arr_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor).item()

    # CCPP dataset range: 420 – 496 MW
    clamped    = max(420.0, min(496.0, output))
    pct        = (clamped - 420.0) / (496.0 - 420.0) * 100

    bar_filled = int(pct / 5)
    bar_empty  = 20 - bar_filled
    bar_str    = "█" * bar_filled + "░" * bar_empty
    load_label = "High ⚡" if pct > 66 else ("Medium ⚡" if pct > 33 else "Low 🔋")

    result = (
        f"## {clamped:.2f} MW\n\n"
        f"`{bar_str}` {pct:.1f}% of rated range\n\n"
        f"---\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Raw prediction | `{output:.4f} MW` |\n"
        f"| Operating range | `420 – 496 MW` |\n"
        f"| Load level | `{load_label}` |\n"
    )
    return result

# ── 4. Theme — industrial dark teal ──────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#080d0e",
    body_background_fill_dark="#080d0e",
    block_background_fill="#0d1517",
    block_background_fill_dark="#0d1517",
    block_border_color="#1a2e32",
    block_border_color_dark="#1a2e32",
    block_label_text_color="#5a8a8f",
    block_label_text_color_dark="#5a8a8f",
    input_background_fill="#111e21",
    input_background_fill_dark="#111e21",
    input_border_color="#1e3a3f",
    input_border_color_dark="#1e3a3f",
    button_primary_background_fill="#0d9488",
    button_primary_background_fill_hover="#0f766e",
    button_primary_text_color="#ffffff",
    slider_color="#0d9488",
    body_text_color="#c8dde0",
    body_text_color_dark="#c8dde0",
)

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto; }

.pg-header {
    background: linear-gradient(135deg, #071012 0%, #0a1c20 60%, #071315 100%);
    border: 1px solid #1a3a40;
    border-radius: 14px;
    padding: 30px 36px 26px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.pg-header::before {
    content: '⚡';
    position: absolute;
    right: 36px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.06;
}
.pg-header h1 {
    font-size: 1.9rem; font-weight: 800;
    color: #2dd4bf;
    letter-spacing: -0.6px;
    margin: 0 0 4px 0;
}
.pg-header .subtitle { color: #3d686e; font-size: 0.87rem; margin: 0 0 14px 0; }
.pill {
    display: inline-block;
    background: #071215; border: 1px solid #1a4a52;
    color: #0d9488;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 12px; border-radius: 999px; margin-right: 8px;
}

.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 20px;
}
.stat-card {
    background: #0d1517; border: 1px solid #1a2e32;
    border-radius: 10px; padding: 14px 16px; text-align: center;
}
.stat-card .val {
    font-size: 1.25rem; font-weight: 800;
    color: #2dd4bf;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1; margin-bottom: 4px;
}
.stat-card .lbl {
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #2a5a60;
}

.eyebrow {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #2a5a60;
    border-left: 3px solid #0d9488; padding-left: 10px;
    margin: 24px 0 12px;
}

.predict-btn {
    margin-top: 24px !important;
    height: 56px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

.pg-footer {
    text-align: center; color: #1e3a3f;
    font-size: 0.76rem;
    margin-top: 20px; padding-top: 16px;
    border-top: 1px solid #111e21;
}
.pg-footer strong { color: #0d9488; }
"""

HEADER = """
<div class="pg-header">
  <h1>GridSync</h1>
  <p class="subtitle">Combined Cycle Power Plant — Net Output Predictor</p>
  <span class="pill">4 Environmental Inputs</span>
  <span class="pill">Regression ANN</span>
  <span class="pill">PyTorch · CCPP Dataset</span>
</div>
"""

STATS = """
<div class="stat-grid">
  <div class="stat-card"><div class="val">9,568</div><div class="lbl">Training Samples</div></div>
  <div class="stat-card"><div class="val">420–496</div><div class="lbl">Output Range (MW)</div></div>
  <div class="stat-card"><div class="val">4→6→6→1</div><div class="lbl">ANN Architecture</div></div>
  <div class="stat-card"><div class="val">MSE</div><div class="lbl">Loss Function</div></div>
</div>
"""

FOOTER = """
<div class="pg-footer">
  Built by <strong>Karthika Krishna M</strong> &nbsp;·&nbsp;
  ANN: Linear(4→6→6→1) &nbsp;·&nbsp;
  CCPP Dataset (UCI ML Repository) &nbsp;·&nbsp;
  Anna University, Tirunelveli
</div>
"""

# ── 5. Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=CSS, title="GridSync — Power Plant Predictor") as demo:

    gr.HTML(HEADER)
    gr.HTML(STATS)

    gr.HTML('<div class="eyebrow">Environmental Conditions</div>')

    with gr.Row():
        at_slider = gr.Slider(
            minimum=1.81, maximum=37.11, value=19.65,
            label="AT — Ambient Temperature (°C)",
            info="Avg 19.65 °C  |  Range: 1.81 – 37.11",
        )
        v_slider = gr.Slider(
            minimum=25.36, maximum=81.56, value=54.31,
            label="V — Exhaust Vacuum (cm Hg)",
            info="Avg 54.31  |  Range: 25.36 – 81.56",
        )

    with gr.Row():
        ap_slider = gr.Slider(
            minimum=992.89, maximum=1033.30, value=1013.26,
            label="AP — Ambient Pressure (mbar)",
            info="Avg 1013.26 mbar  |  Range: 992.89 – 1033.30",
        )
        rh_slider = gr.Slider(
            minimum=25.56, maximum=100.16, value=73.31,
            label="RH — Relative Humidity (%)",
            info="Avg 73.31 %  |  Range: 25.56 – 100.16",
        )

    btn = gr.Button(
        "  Predict Net Power Output",
        variant="primary",
        elem_classes=["predict-btn"],
    )

    gr.HTML('<div class="eyebrow">Prediction Result</div>')

    output = gr.Markdown(
        value="> Adjust the sliders above and click **Predict** to estimate net power output.",
    )

    btn.click(
        fn=predict,
        inputs=[at_slider, v_slider, ap_slider, rh_slider],
        outputs=output,
    )

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch()