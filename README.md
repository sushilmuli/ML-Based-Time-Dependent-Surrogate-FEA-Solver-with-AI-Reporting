# ML-Based-Time-Dependent-Surrogate-FEA-Solver-with-AI-Reporting
Built a hybrid ML-based surrogate model for beam FEA to predict stress, deflection, and plastic strain, combined with physics-based time evolution. Integrated Streamlit UI and automated AI-driven engineering report generation with PDF export.

Overview:
This project combines surrogate modeling (Random Forest) with engineering physics to provide fast and reliable structural predictions.
Unlike traditional FEA solvers, this tool:
-Generates time-dependent response curves
-Provides AI-based engineering insights
-Exports automated PDF reports

Key Features
•	ML Surrogate Model for predicting:
1.	Deflection
2.	von Mises stress
3.	Plastic strain
•	Hybrid Modelling Approach
1.	ML → predicts maximum response
2.	Physics → governs time evolution
•	Time-Dependent Simulation
1.	Deflection vs Time
2.	Stress vs Time
3.	Plastic Strain vs Time
•	Material Modelling
1.	Elastic + Plastic behaviour
2.	Yield & Ultimate strength-based failure checks
•	Boundary Conditions
1.	Cantilever beam
2.	Simply supported beam
•	Interactive UI
1.	Built using Streamlit
2.	Real-time parameter tuning
•	Automated Reporting
1.	PDF report generation
2.	Includes plots + engineering summary
•	AI Integration
1.	Generates:
2.	Engineering interpretation
3.	Failure assessment
4.	Design recommendations

Tech Stack:
1.	Python
2.	Streamlit
3.	Scikit-learn (Random Forest)
4.	NumPy / Pandas
5.	Matplotlib
6.	ReportLab (PDF generation)
7.	OpenAI API (AI reporting)

How to run:
1. Install dependencies
2. Set API Key (optional): export OPENAI_API_KEY=your_api_key  
3. Run application: streamlit run app.py

Key Takeaway:
1. ML predicts WHAT happens
2. Physics explains HOW it happens

Author
Sushil Muli
FEA Engineer | PG in AIML-Deep Learning
