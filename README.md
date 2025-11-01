# 🌾 CropShield AI  
### AI-Powered Plant Disease Detection, Smart Recommendations & Impact Tracking  

---

## 🧠 Overview
**CropShield AI** is an AI-driven plant health assistant designed to help farmers and researchers **diagnose crop diseases**, **receive actionable treatment advice**, and **track environmental impact** — all through a simple web interface built with **Streamlit**.

It combines **deep learning–based image diagnosis**, **climate data integration**, and **recommendation logic** to provide fast, accessible, and explainable plant health insights.

---

## 🚀 Features

✅ **Image-Based Diagnosis**  
Upload a photo of a plant leaf — the app identifies diseases and reports confidence levels.  

✅ **Smart Recommendations**  
Get customized chemical and organic treatment suggestions, dosage guides, and climate-adjusted prevention steps.  

✅ **Weather-Aware Diagnosis**  
Optionally input or auto-fetch local weather (temperature, humidity, rainfall, pH/NPK).  

✅ **Explainable AI (GradCAM)**  
View a simple overlay showing which parts of the leaf image influenced the AI’s decision.  

✅ **Impact Tracking**  
Monitor pesticide reduction, water savings, and yield preservation over time through charts and metrics.  

✅ **Multi-Language Support**  
Switch between English and Hindi for easy accessibility.  

✅ **Offline-Ready Frontend**  
Fully local Streamlit interface, no internet dependency for inference (uses mock API in prototype).

---

## 🧩 Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| **Frontend** | Streamlit, Plotly/Altair, Tailwind-style CSS |
| **Mock Backend** | Python functions simulating API responses (`utils/mock_api.py`) |
| **Visualization** | Plotly / Altair for charts |
| **Report Generation** | FPDF / Streamlit download button |
| **Styling** | Custom CSS (`utils/style.css`) |
| **Data Management** | Streamlit session state |

---

## 📂 Project Structure
