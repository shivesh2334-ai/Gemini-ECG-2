import streamlit as st
from PIL import Image
import google.generativeai as genai
import anthropic
from groq import Groq
import base64
import io

# --- Helper: Image to Base64 ---
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Model API Wrappers ---
def call_gemini(api_key, image, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
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

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Multi-Model ECG Analysis", layout="wide", page_icon="🫀")
    
    st.markdown("""
        <style>
        .step-header {font-size: 1.3rem; color: #1565C0; font-weight: bold; margin-top: 15px;}
        .report-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1565C0;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🫀 Hybrid AI ECG Interpretation")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox("Select AI Model", ["Gemini 1.5 Pro", "Claude 3.5 Sonnet", "Llama 3.2 Vision (Groq)"])
        
        api_key = ""
        if "Gemini" in model_choice:
            api_key = st.text_input("Gemini API Key", type="password")
        elif "Claude" in model_choice:
            api_key = st.text_input("Anthropic API Key", type="password")
        elif "Llama" in model_choice:
            api_key = st.text_input("Groq API Key", type="password")

        st.divider()
        uploaded_file = st.file_uploader("Upload ECG Image", type=['png', 'jpg', 'jpeg'])
        clinical_context = st.text_area("Clinical Context", "e.g., 60M, Hypertensive, Chest Pain")

    if not uploaded_file:
        st.info("👋 Please upload an ECG image to begin.")
        st.stop()

    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image, use_container_width=True, caption="ECG Trace")

    with col2:
        findings = {"Context": clinical_context}
        
        st.header("📝 Manual Findings Checklist")
        
        # --- Simplified Steps 1-6 ---
        st.markdown('<div class="step-header">Basic Parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            findings['Rate'] = st.number_input("Heart Rate (bpm)", value=75)
            findings['Rhythm'] = st.selectbox("Rhythm", ["Sinus Rhythm", "Sinus Tachycardia", "Sinus Bradycardia", "Atrial Fibrillation", "Atrial Flutter", "VT"])
            findings['Axis'] = st.selectbox("Axis", ["Normal", "LAD", "RAD", "Extreme Axis"])
        with c2:
            findings['PR'] = st.number_input("PR Interval (ms)", value=160)
            findings['QRS'] = st.number_input("QRS Duration (ms)", value=90)
            findings['QTc'] = st.number_input("QTc (ms)", value=420)

        # --- STEP 7: Detailed ST/T Analysis ---
        st.markdown('<div class="step-header">Step 7: ST Segment & T Waves</div>', unsafe_allow_html=True)
        
        st_options = ["Normal", "ST Elevation", "ST Depression", "T Wave Inversion", "Hyperacute T Waves", "Pathological Q Waves"]
        st_findings_list = st.multiselect("Select observed morphologies:", st_options, default=["Normal"])
        
        detailed_st_text = []
        
        if "Normal" not in st_findings_list and len(st_findings_list) > 0:
            leads_list = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            st.write("### 📍 Localization")
            for abnormality in st_findings_list:
                if abnormality != "Normal":
                    selected_leads = st.multiselect(f"Which leads show **{abnormality}**?", leads_list, key=abnormality)
                    if selected_leads:
                        detailed_st_text.append(f"{abnormality} in leads {', '.join(selected_leads)}")
            
            findings['ST_Status'] = "; ".join(detailed_st_text) if detailed_st_text else "Abnormalities noted but leads not specified."
        else:
            findings['ST_Status'] = "Normal ST segments and T waves throughout."

        st.info(f"**Current ST Summary:** {findings['ST_Status']}")

        # --- AI Execution ---
        st.divider()
        if st.button(f"Generate Diagnosis ({model_choice})"):
            if not api_key:
                st.error(f"Please provide the API Key for {model_choice}.")
            else:
                prompt = f"""
                You are an expert Consultant Cardiologist.
                
                Patient Context: {findings['Context']}
                
                I have performed a manual analysis of the image with these findings:
                - Rate: {findings['Rate']} bpm
                - Rhythm: {findings['Rhythm']}
                - Intervals: PR {findings['PR']}ms, QRS {findings['QRS']}ms, QTc {findings['QTc']}ms
                - Axis: {findings['Axis']}
                - ST/T Changes: {findings['ST_Status']}
                
                Please analyze the attached image and cross-reference with my findings.
                1. CONFIRM: Do you see the ST/T changes in the leads I described?
                2. DIAGNOSE: Provide the ECG diagnosis (e.g., Anterolateral STEMI, LVH with Strain).
                3. PLAN: Suggest immediate clinical next steps.
                """
                
                with st.spinner(f"Sending data to {model_choice}..."):
                    try:
                        result = ""
                        if "Gemini" in model_choice:
                            result = call_gemini(api_key, image, prompt)
                        elif "Claude" in model_choice:
                            result = call_claude(api_key, image, prompt)
                        elif "Llama" in model_choice:
                            result = call_llama(api_key, image, prompt)
                        
                        st.markdown('<div class="report-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🤖 Analysis by {model_choice}")
                        st.markdown(result)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")

if __name__ == "__main__":
    main()
