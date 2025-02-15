# TB Sahayak

TB Sahayak is a web application designed to assist in the early detection and awareness of Tuberculosis (TB). The platform allows users to upload chest X-rays for analysis using an AI model, provides symptom check functionality, and helps locate nearby healthcare facilities.

## Features

### 🔍 AI-Powered X-ray Analysis
- Upload a chest X-ray image to get a preliminary assessment.
- Interactive loading animation while processing.

### 🏥 Find Nearby Healthcare Facilities
- Locate hospitals, clinics, and pharmacies near your location.
- Get directions to the nearest healthcare facility.

### 📋 Symptom Checker
- Answer a few questions to assess symptoms.
- Get guidance on potential next steps.

### 🗺️ IITGN Interactive Map
- A campus-wide interactive map for better navigation.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Backend:** Flask (Python)
- **AI Model:** Deep Learning-based X-ray classification
- **Database:** SQLite (if applicable)
- **APIs Used:** Google Maps API for location services
- **Deployment:** Railway

## Deployment
The project is deployed on **Railway**.

🔗 **Live Version:** [TB Sahayak Deployment](https://tb-sahayak-production.up.railway.app/)

**⚠️ Note:** The deployed version may experience downtime due to free-tier limitations on CPU and RAM.

## How to Run Locally

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- Required dependencies from `requirements.txt`

### Installation Steps
```sh
# Clone the repository
git clone https://github.com/your-repo-url.git
cd tb-sahayak

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

## Future Enhancements
- Improve AI model accuracy
- Add a multilingual interface
- Expand healthcare facility database

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

---
🚀 **Developed with passion for public health and AI-driven solutions!**

