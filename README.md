# 🎨 AI Sketch Creator  

> Convert any photo into a **realistic pencil sketch** using AI-powered image processing.  
> Built with **Flask, OpenCV, Pillow, and NumPy** — simple, fast, and visually stunning.  

---

## 🧠 Project Overview  

**AI Sketch Creator** transforms your uploaded images into beautiful pencil sketches.  
You can easily adjust:
- ✏️ **Pen Thickness** — control stroke intensity  
- 🌈 **Contrast** — fine-tune the sketch depth  

All processing happens directly in Python using OpenCV filters and NumPy math magic.  

---

## 🚀 Tech Stack  

| Technology | Purpose |
|-------------|----------|
| 🐍 **Python** | Core programming language |
| 🌐 **Flask** | Web framework |
| 🧩 **OpenCV** | Image processing & sketch effect |
| 🖼️ **Pillow (PIL)** | Image manipulation |
| 🔢 **NumPy** | Array-based computation |

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/AI-Sketch-Creator.git
cd AI-Sketch-Creator
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```
python -m venv venv
venv\Scripts\activate       # For Windows
# source venv/bin/activate  # For macOS/Linux
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```
python app.py
```
Visit: http://127.0.0.1:5000/
---
## 🖼️ Features

✅ Upload any photo (JPG, PNG)
✅ Adjust Pen Thickness & Contrast
✅ Realistic Pencil Sketch Output
✅ Clean, Glassmorphism UI
✅ Download or regenerate sketches easily

## 📂 Project Structure
```
AI-Sketch-Creator/
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
└── README.md
```
| Original                                           | Sketch                                                    |
| -------------------------------------------------- | --------------------------------------------------------- |
| <img src="static/uploads/sample.jpg" width="250"/> | <img src="static/uploads/sketch_sample.jpg" width="250"/> |

## 👨‍💻 Author

Vineet Sharma
```
Data Scientist | Full Stack Developer | AI Enthusiast
```
## 🏷️ License
This project is open-source under the MIT License — free to use, modify, and share.
