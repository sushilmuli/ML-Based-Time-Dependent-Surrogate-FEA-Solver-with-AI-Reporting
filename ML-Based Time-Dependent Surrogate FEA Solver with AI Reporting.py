# =================  (FULL CODE - ML SURROGATE VERSION) =================
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import joblib

# Report
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ================= OpenAI =================
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    USE_LLM = True
else:
    USE_LLM = False
    st.warning("API Key not found. Running without AI report.")

st.set_page_config(page_title="ML Surrogate FEA Solver", layout="wide")
st.title("  ML-Based Surrogate FEA Solver")

# ================= MODEL =================
MODEL_PATH = "beam_ml_model.pkl"

def train_model():
    N = 5000
    features = []
    targets = []

    for _ in range(N):
        width = np.random.uniform(10, 100)
        depth = np.random.uniform(2, 10)
        L = np.random.uniform(50, 200)
        E = 210e3
        yield_strength = 250
        load = np.random.uniform(500, 5000)
        bc = np.random.choice([0, 1])

        A = width * depth
        I = (width * depth**3)/12

        if bc == 0:
            max_disp = (load * L**3)/(3*E*I)
            M = load * L
        else:
            max_disp = (load * L**3)/(48*E*I)
            M = load * L / 4

        y = depth/2
        stress = (M*y)/I

        tau = (1.5 * load) / A
        von_mises = np.sqrt(stress**2 + 3*tau**2)

        plastic_strain = max(0, (von_mises - yield_strength)/E)

        # Load position definition
        if bc == 0:   # Cantilever
            load_pos = 1.0   # end of beam
        else:         # Simply Supported
            load_pos = 0.5   # center of beam
        features.append([width, depth, L, load, bc, load_pos])

        targets.append([stress, max_disp, von_mises, tau, plastic_strain])

    X = pd.DataFrame(features)
    y = pd.DataFrame(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=100)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    with st.spinner("Training ML model..."):
        model = train_model()

# ================= INPUTS ====================================================================

st.sidebar.header("Material")
E = st.sidebar.number_input("Young’s Modulus (MPa)", value=210e3)
nu = st.sidebar.number_input("Poisson Ratio", value=0.3)
rho = st.sidebar.number_input("Density (kg/mm³)", value=7.85e-9, format='%.2e')

yield_strength = st.sidebar.number_input("Yield Strength (MPa)", value=250)
ultimate_strength = st.sidebar.number_input("Ultimate Strength (MPa)", value=450)
ultimate_plastic_strain = st.sidebar.number_input("Ultimate Plastic Strain", value=0.15)

st.sidebar.header("Geometry")
width = st.sidebar.number_input("Width (mm)", value=40)
depth = st.sidebar.number_input("Depth (mm)", value=5)
L = st.sidebar.number_input("Length (mm)", value=80)

# Mass
A = width * depth
volume = A * L
mass = rho * volume
I = (width * depth**3)/12
st.sidebar.write(f"Mass: {mass:.4f} kg")

##Loading
st.sidebar.header("Loading")
Fz = st.sidebar.number_input("Vertical Load Fz (N)", value=1000.0)

#Boundary condition
st.sidebar.header("Boundary Condition")
bc = st.sidebar.selectbox("Support Type", ["Cantilever", "Simply Supported"])
bc_val = 0 if bc == "Cantilever" else 1


# ================= TIME SETTINGS =================
st.sidebar.header("Time Parameters")
total_time = st.sidebar.number_input("Total Time (ms)", value=100.0)
time_steps = st.sidebar.number_input("Time Steps", min_value=10, value=50)

num_elements = 50

# ================= SINGLE ML PREDICTION ======================================================================
if bc_val == 0:
    load_pos = 1.0
else:
    load_pos = 0.5

features = np.array([[width, depth, L, Fz, bc_val, load_pos]])
pred = model.predict(features)[0]

stress, max_disp, max_vm, tau, max_plastic = pred

# ================= TIME ARRAY ================================================================================
time = np.linspace(0, total_time, time_steps)
time_factor = time / max(time)

# ================= DEFLECTION & STRESS (LINEAR) =================
deflection_time = max_disp * time_factor
vm_time = max_vm * time_factor

# ================= PLASTIC STRAIN (NONLINEAR) =================
plastic_time = []

for t_factor in time_factor:
    vm_t = max_vm * t_factor

    if vm_t > yield_strength:
        plastic_t = (vm_t - yield_strength) / E   #
    else:
        plastic_t = 0

    plastic_time.append(plastic_t)

# ================= SAFETY ====================================================================================
failure_flag = False

max_vm_val = max(vm_time)
max_plastic_val = max(plastic_time)

if max_plastic_val >= ultimate_plastic_strain:
    st.error("❌ Failure: Plastic strain exceeded")

elif max_vm_val > ultimate_strength:
    st.error("❌ Failure: Ultimate strength exceeded")  

elif max_vm_val > yield_strength:
    st.warning("⚠️ Plastic deformation")

else:
    st.success("✅ Elastic")

# ================= RESULTS =================
st.subheader("Results")

c1, c2, c3 = st.columns(3)

c1.metric("Max Deflection", f"{max(deflection_time):.6f}")
c2.metric("Max von Mises", f"{max(vm_time):.2e}")
c3.metric("Plastic Strain", f"{max(plastic_time):.2e}")

# ================= PLOTS =================
st.subheader("Deflection vs Time (ms)")
st.line_chart(pd.DataFrame({"Time": time, "Deflection": deflection_time}).set_index("Time"))

st.subheader("von Mises vs Time (ms)")
st.line_chart(pd.DataFrame({"Time": time, "Stress": vm_time}).set_index("Time"))

st.subheader("Plastic Strain vs Time (ms)")
st.line_chart(pd.DataFrame({"Time": time, "Plastic": plastic_time}).set_index("Time"))

# ================= SAVE PLOTS FOR PDF =================
def generate_plot_images():
    images = {}

    # Deflection plot
    fig1, ax1 = plt.subplots()
    ax1.plot(time, deflection_time)
    ax1.set_title("Deflection")
    img1 = BytesIO()
    fig1.savefig(img1, format='png')
    img1.seek(0)
    images["deflection"] = img1
    plt.close(fig1)

    # von Mises plot
    fig2, ax2 = plt.subplots()
    ax2.plot(time, vm_time)
    ax2.set_title("von Mises Stress")
    img2 = BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    images["vm"] = img2
    plt.close(fig2)

    # Plastic strain plot
    fig3, ax3 = plt.subplots()
    ax3.plot(time,plastic_time)
    ax3.set_title("Plastic Strain")
    img3 = BytesIO()
    fig3.savefig(img3, format='png')
    img3.seek(0)
    images["plastic"] = img3
    plt.close(fig3)

    return images

# ================= FORMAT AI REPORT =================
def format_ai_report(report):
    sections = report.split("###")
    formatted = []

    for sec in sections:
        sec = sec.strip()
        if sec:
            lines = sec.split("\n")
            title = lines[0]
            content = "<br/>".join(lines[1:])
            formatted.append((title, content))

    return formatted    

# ================= REPORT (UNCHANGED LOGIC) =================
def create_full_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    # Generate images
    images = generate_plot_images()

    elements.append(Paragraph("<b>ML-Based FEA Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    # ================= MATERIAL TABLE =================
    material_data = [
        ["Property", "Value"],
        ["Young's Modulus", str(E)],
        ["Poisson Ratio", str(nu)],
        ["Density", str(rho)],
        ["Yield Strength", str(yield_strength)],
        ["Ultimate Strength", str(ultimate_strength)]
    ]

    table = Table(material_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(Paragraph("<b>Material Properties</b>", styles["Heading2"]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # ================= GEOMETRY =================
    elements.append(Paragraph("<b>Geometry</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Width: {width}, Depth: {depth}, Length: {L}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # ================= RESULTS =================
    elements.append(Paragraph("<b>Results</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Max Deflection: {max_disp}", styles["Normal"]))
    elements.append(Paragraph(f"Max von Mises: {max(vm_time)}", styles["Normal"]))
    elements.append(Paragraph(f"Max Plastic Strain: {max(plastic_time)}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # ================= ADD PLOTS =================
    elements.append(Paragraph("<b>Deflection Plot</b>", styles["Heading2"]))
    elements.append(Image(images["deflection"], width=400, height=250))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>von Mises Stress Plot</b>", styles["Heading2"]))
    elements.append(Image(images["vm"], width=400, height=250))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Plastic Strain Plot</b>", styles["Heading2"]))
    elements.append(Image(images["plastic"], width=400, height=250))
    elements.append(Spacer(1, 10))


#==================== AI REPORT =================
    elements.append(Paragraph("<b>AI Engineering Report</b>", styles["Heading1"]))
    elements.append(Spacer(1, 10))

    if report:
        sections = format_ai_report(report)

        for title, content in sections:
            elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            elements.append(Paragraph(content, styles["Normal"]))
            elements.append(Spacer(1, 10))
 

    doc.build(elements)
    buffer.seek(0)
    return buffer### nothing will plot after return statement

# ================= AI REPORT =================
if st.button("Generate AI Report"):

    prompt = f"""
    You are a senior CAE crash and structural analyst.

    Evaluate the following FEA results:

    Beam Type: {bc}

    Material:
    E = {E}, ν = {nu}, ρ = {rho}
    Yield Strength: {yield_strength}
    Ultimate Strength: {ultimate_strength}
    Ultimate Plastic Strain: {ultimate_plastic_strain}

    Geometry:
    Width: {width}, Depth: {depth}, Length: {L}

    Loading: Fz = {Fz} N

    Results:
    Max Deflection = {max_disp}
    Max von Mises stress = {max(vm_time)}
    Plastic Strain = {max(plastic_time)}

    Provide output EXACTLY in this format:

    ### Engineering Interpretation
    - point
    - point
    - point
    - point

    ### Failure Assessment
    - point
    - point
    - point
    - point
    - point

    ### Nonlinearity Insights
    - point
    - point

    ### Design Risks
    - point
    - point
    - point
    - point
    - point

    ### Recommendations
    - point
    - point
    - point
    - point
    - point

    Keep answers concise and professional.
    """

    if USE_LLM:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        report = response.choices[0].message.content
    else:
        report = "Basic report generated without AI."

    st.subheader("Report")
    st.markdown(report)

    # FULL CAE PDF EXPORT
    pdf_file = create_full_pdf(report)

    st.download_button(
        label="Download Full FEA Report (PDF)",
        data=pdf_file,
        file_name="FEA_Report.pdf",
        mime="application/pdf"
    )