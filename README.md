﻿# TB_Sahayak

TB Sahayak is a **web-based tuberculosis (TB) diagnosis assistant** that allows users to **upload chest X-ray images** for **AI-based TB detection** and **locate nearby healthcare facilities** using Google Maps.

## 🚀 Features

### 🔍 AI-Based TB Detection
- Users can **upload a chest X-ray image**.
- The system analyzes the image and predicts whether **TB is detected or not**.
- Displays a **loading animation** while processing.

### 📍 Google Maps Integration
- Searches **hospitals, clinics, doctors, and pharmacies** nearby.
- Uses **live location access** to show the closest healthcare facilities.
- Users can **get directions** to a selected place.

### 🎨 UI & UX
- **Bootstrap-based responsive design**.
- **Smooth animations & Poppins font** for a clean look.
- **Interactive loading animation** (spinner + typewriter effect for "Loading...").

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Flask (Python)
- **APIs**:
  - Google Maps API (Places, Directions, Routes)
  - Custom AI Model API for X-ray analysis

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/tb-sahayak.git
   cd tb-sahayak
   ```
2. **Set up a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   - Add your **Google Maps API key** to a `.env` file:
     ```
     GOOGLE_MAPS_API_KEY=your_api_key_here
     ```

5. **Run the Flask app**
   ```sh
   flask run
   ```
   The application will run on `http://127.0.0.1:5000/`

## 📌 How to Use

1. **Upload an X-ray**: Click the upload button and select a chest X-ray image.
2. **Wait for AI Prediction**: The system will process the image and display the result.
3. **Find Nearby Healthcare**: Click on "Find Hospitals" to see available healthcare centers.
4. **Get Directions**: Select a location and get navigation assistance.

## 🐞 Common Issues & Fixes

### ❌ X-ray Preview Not Loading
- Ensure the `<img>` tag has a valid `src` after file selection.
- Use JavaScript's `FileReader` to display the selected image.

### ❌ Google Maps Not Displaying
- Check if the **API key is correct** and enabled for `Maps JavaScript API`, `Places API`, and `Directions API`.
- Ensure there are **no CORS or billing issues** on the Google Cloud Console.

## 📜 License

This project is licensed under the **MIT License**.

---
Made with ❤️ for AI-driven healthcare assistance! 🚀
