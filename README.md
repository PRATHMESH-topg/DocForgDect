# 🧠 DocForgDect — Document Forgery Detection System

> A Deep Learning–based intelligent system that detects **forged** and **original** documents with visual analytics using a **ResNet50 transfer learning model** and a **FastAPI backend** integrated with an interactive **HTML + CSS frontend** featuring a real-time **Pie Chart visualization**.

---

## 🌟 Overview

Document forgery is a major concern in legal, financial, and identity verification systems.  
**DocForgDect** aims to **automate document authenticity verification** using **AI-driven image analysis**.  

This system:
- Processes uploaded document images  
- Classifies them as **“Original”** or **“Forged”**  
- Displays results visually as a **Pie Chart (Original vs Forged percentage)**  
- Provides a simple, responsive **HTML interface**  
- Runs on a **FastAPI backend** powered by a fine-tuned **ResNet50** model  

---

## 🧩 Project Architecture


## 🚀 Features

✅ **AI-based Forgery Detection:** Uses a deep ResNet50 network to classify document authenticity.  
✅ **Accurate Predictions:** Model fine-tuned with strong augmentations to achieve over **80% accuracy**.  
✅ **Visual Representation:** Displays prediction results as a **dynamic Pie Chart** using Chart.js.  
✅ **Interactive Web UI:** HTML/CSS interface for file upload, loading animation, and reset functionality.  
✅ **Cross-Origin Ready:** Backend supports **CORS** for frontend API access.  
✅ **Auto Model Saving:** Automatically saves the **best-performing model** during training.  

---

💻 Frontend Demo

The frontend folder contains a simple and clean HTML page for demonstration.

🔹 Features:

Upload any image (document)

Click “Test Document”

See a loading animation while model processes

Display result as a Pie Chart (Original vs Forged %)

Reset button clears chart and file input
