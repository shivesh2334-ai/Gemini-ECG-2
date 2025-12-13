# 🫀 Multi-Model AI ECG Assistant

An advanced ECG analysis tool that combines human-led checklist workflows with diagnostics from state-of-the-art AI Vision models (**Gemini 1.5 Pro, Claude 3.5 Sonnet, and Llama 3.2 Vision**).

## 🚀 Features
*   **Manual Workflow:** Digital checklist for Rate, Rhythm, Axis, Intervals, and ST-T changes.
*   **Specific Localization:** Granular selector for ST changes (e.g., "ST Elevation in V1, V2").
*   **Multi-Model Support:** Switch between Google, Anthropic, and Meta models.
*   **Benchmarking Script:** Includes python script to test accuracy (F1 Score) against PTB-XL dataset.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ECG-AI-Assistant.git
    cd ECG-AI-Assistant
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Option 1: Run Locally
```bash
streamlit run ecg_app.py
