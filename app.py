
import streamlit as st
from PIL import Image
import math
import google.generativeai as genai
import anthropic
from groq import Groq
import base64
import io

# --- Helper: Image to Base64 (for Claude/Llama) ---
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Model Handlers ---

def call_gemini(api_key, image, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro') # Using Pro for better medical reasoning
    response = model.generate_content([prompt, image])
    return response.text

def call_claude(api_key, image, prompt):
    client = anthropic.Anthropic(api_key=api_key)
    base64_image = encode_image(image)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return message.content[0].text

def call_llama(api_key, image, prompt):
    # Using Groq for Llama 3.2 Vision (11B or 90B)
    client = Groq(api_key=api_key)
    base64_image = encode_image(image)
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        model="llama-3.2-11b-vision-preview",
    )
    return chat_completion.choices[0].message.content

# --- Main Application ---
def main():
    st.set_page_config(page_title="Multi-Model ECG Analysis", layout="wide", page_icon="🫀")
    
    st.markdown("""
        <style>
        .step-header {font-size: 1.3rem; color: #1565C0; font-weight: bold; margin-top: 15px;}
        .report-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🫀 AI-Enhanced ECG Interpretation")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_choice = st.selectbox("Choose AI Model", ["Gemini 1.5 Pro", "Claude 3.5 Sonnet", "Llama 3.2 Vision (Groq)"])
        
        api_key = ""
        if "Gemini" in model_choice:
            api_key = st.text_input("Gemini API Key", type="password")
        elif "Claude" in model_choice:
            api_key = st.text_input("Anthropic API Key", type="password")
        elif "Llama" in model_choice:
            api_key = st.text_input("Groq API Key", type="password")

        st.divider()
        uploaded_file = st.file_uploader("Upload ECG Image", type=['png', 'jpg', 'jpeg'])
        clinical_context = st.text_area("Clinical Context", "55yo Male, Chest Pain")

    if not uploaded_file:
        st.info("Upload an ECG to start.")
        st.stop()

    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, use_container_width=True)

    with col2:
        findings = {"Context": clinical_context}
        
        # --- Manual Checklist Steps ---
        
        # Steps 0-6 (Simplified for brevity, assuming previous logic exists)
        st.markdown('<div class="step-header">Basic Measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            findings['Rate'] = st.number_input("Heart Rate (bpm)", value=75)
            findings['Rhythm'] = st.selectbox("Rhythm", ["Sinus", "Afib", "A-Flutter", "VT"])
            findings['Axis'] = st.selectbox("Axis", ["Normal", "LAD", "RAD", "Extreme"])
        with c2:
            findings['PR'] = st.number_input("PR Interval (ms)", value=160)
            findings['QRS'] = st.number_input("QRS Duration (ms)", value=90)
            findings['QTc'] = st.number_input("QTc (ms)", value=420)

        # --- STEP 7: Updated ST/T Changes ---
        st.markdown('<div class="step-header">Step 7: ST Segment & T Waves</div>', unsafe_allow_html=True)
        
        st_options = ["Normal", "ST Elevation", "ST Depression", "T Inversion", "Hyperacute T"]
        st_finding = st.multiselect("Select Morphological Changes:", st_options, default=["Normal"])
        
        if "Normal" not in st_finding and len(st_finding) > 0:
            leads_list = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            affected_leads = st.multiselect(f"Select Leads showing {', '.join(st_finding)}:", leads_list)
            findings['ST_Status'] = f"{', '.join(st_finding)} in leads {', '.join(affected_leads)}"
        else:
            findings['ST_Status'] = "Normal ST/T segments"

        # --- AI Execution ---
        st.divider()
        if st.button(f"Analyze with {model_choice}"):
            if not api_key:
                st.error("Please provide an API Key.")
            else:
                prompt = f"""
                Act as an expert Cardiologist. Analyze the attached ECG image and my manual findings.
                
                Patient: {findings['Context']}
                Manual Findings:
                - Rate/Rhythm: {findings['Rate']} bpm, {findings['Rhythm']}
                - Intervals: PR {findings['PR']}, QRS {findings['QRS']}, QTc {findings['QTc']}
                - Axis: {findings['Axis']}
                - ST/T Changes: {findings['ST_Status']}
                
                Task:
                1. Verify my manual ST/T findings against the image.
                2. Provide a final diagnosis (e.g., STEMI, NSTEMI, Normal, LVH).
                3. Recommend next steps.
                """
                
                with st.spinner("Consulting AI Model..."):
                    try:
                        result = ""
                        if "Gemini" in model_choice:
                            result = call_gemini(api_key, image, prompt)
                        elif "Claude" in model_choice:
                            result = call_claude(api_key, image, prompt)
                        elif "Llama" in model_choice:
                            result = call_llama(api_key, image, prompt)
                        
                        st.markdown('<div class="report-box">', unsafe_allow_html=True)
                        st.markdown(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")

if __name__ == "__main__":
    main()
