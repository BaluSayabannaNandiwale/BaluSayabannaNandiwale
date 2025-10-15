# Moodify – Intelligent Mood Detection System

[![GitHub](https://img.shields.io/github/license/BaluSayabannaNandiwale/emotion-detection-project)](https://github.com/BaluSayabannaNandiwale/emotion-detection-project)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://www.tensorflow.org/)

A comprehensive web application that intelligently detects user emotions from face images, speech recordings, and text input using pre-trained machine learning models.

![Moodify Banner](https://img.shields.io/badge/Moodify-Emotion%20Detection-blue?style=for-the-badge&logo=brain&logoColor=white)

## 🌟 Featured In

[![GitHub](https://img.shields.io/badge/GitHub-Project%20Repository-blue?logo=github)](https://github.com/BaluSayabannaNandiwale/emotion-detection-project)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/balusayabannandiwale/)

## 🚀 Demo

![Project Demo](https://raw.githubusercontent.com/BaluSayabannaNandiwale/emotion-detection-project/main/demo/moodify-demo.gif)

## Features

### 🎭 Multi-Modal Emotion Detection
- **Face Emotion Detection**: Upload images or use webcam for real-time facial emotion recognition
- **Speech Emotion Detection**: Upload audio files or record directly for voice emotion analysis  
- **Text Emotion Detection**: Type or paste text for sentiment and emotion analysis

### 🎨 Modern User Interface
- Responsive design that works on desktop and mobile
- Intuitive tab-based navigation between detection modes
- Real-time webcam capture and audio recording
- Interactive charts showing emotion probabilities
- Beautiful animations and visual feedback

### 📊 Analytics Dashboard
- Historical emotion detection data
- Emotion distribution charts
- Confidence score tracking
- Recent predictions table

### 📊 Analytics Dashboard
- Historical emotion detection data
- Emotion distribution charts
- Confidence score tracking
- Recent predictions table

### 🔧 Technical Features
- Flask backend with RESTful API endpoints
- SQLite database for storing prediction history
- Pre-trained ML models (TensorFlow, scikit-learn)
- Audio processing with librosa and MFCC features
- Computer vision with OpenCV for face detection
- Real-time audio visualization

## 📁 Project Structure

```
emotion-project/
├── app_simple.py          # Main Flask application (simplified version)
├── app.py                 # Full Flask application (with model compatibility issues)
├── requirements.txt       # Python dependencies
├── emotion_history.db     # SQLite database (created automatically)
├── model/                 # Pre-trained models directory
│   ├── facialemotionmodel.h5
│   ├── facialemotionmodel.json
│   ├── speech_emotion_model.h5
│   ├── text_emotion.pkl
│   └── history.pkl
├── templates/             # HTML templates
│   ├── index.html         # Main application interface
│   └── analytics.html     # Analytics dashboard
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Application styling
│   ├── js/
│   │   └── app.js         # Frontend JavaScript
│   └── images/
└── README.md              # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Installation

1. **Clone the repository**
```bash
git clone https://github.com/BaluSayabannaNandiwale/emotion-detection-project.git
cd emotion-detection-project
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app_simple.py
```

5. **Access the application**
Open your browser and go to: http://127.0.0.1:5000

### Model Files
Ensure these model files are in the `model/` directory:
- `text_emotion.pkl` ✓ (Working)
- `facialemotionmodel.h5`
- `facialemotionmodel.json`
- `speech_emotion_model.h5`
- `history.pkl`

## Usage Guide

### Face Emotion Detection
1. Click on the "Face" tab
2. **Option A**: Upload an image file
   - Drag & drop an image or click "browse"
   - Supported formats: JPG, PNG, GIF
3. **Option B**: Use webcam
   - Click "Start Webcam"
   - Click "Capture Photo" when ready
   - Click "Stop Webcam" when done
4. Click "Analyze Emotion" to get results

### Speech Emotion Detection
1. Click on the "Speech" tab
2. **Option A**: Upload an audio file
   - Drag & drop an audio file or click "browse"
   - Supported formats: WAV, MP3, M4A
3. **Option B**: Record audio
   - Click "Start Recording"
   - Speak clearly into your microphone
   - Click "Stop Recording" when done
   - Optionally click "Play Recording" to review
4. Click "Analyze Emotion" to get results

### Text Emotion Detection
1. Click on the "Text" tab
2. Type or paste text (up to 1000 characters)
3. Click "Analyze Emotion" to get results
4. Use "Clear" to reset the text area

### View Analytics
- Click the "Analytics" tab in the navigation
- View emotion distribution charts
- See recent prediction history
- Track confidence scores over time

## API Endpoints

### Prediction Endpoints
- `POST /predict_face` - Face emotion detection
- `POST /predict_speech` - Speech emotion detection  
- `POST /predict_text` - Text emotion detection

### Pages
- `GET /` - Main application interface
- `GET /analytics` - Analytics dashboard

## 🌐 API Endpoints

### Prediction Endpoints
- `POST /predict_face` - Face emotion detection
- `POST /predict_speech` - Speech emotion detection  
- `POST /predict_text` - Text emotion detection

### Pages
- `GET /` - Main application interface
- `GET /analytics` - Analytics dashboard

## 🤖 Model Information

### Text Emotion Model
- **Status**: ✅ Working
- **Framework**: scikit-learn Pipeline
- **Features**: CountVectorizer + LogisticRegression
- **Emotions**: anger, disgust, fear, joy, neutral, sadness, shame, surprise

### Face Emotion Model
- **Status**: ⚠️ Demo mode (compatibility issues)
- **Framework**: TensorFlow/Keras CNN
- **Input**: 48x48 grayscale images
- **Emotions**: angry, disgust, fear, happy, neutral, sad, surprise

### Speech Emotion Model
- **Status**: ⚠️ Demo mode (compatibility issues)
- **Framework**: TensorFlow/Keras
- **Features**: MFCC, chroma, mel spectrogram, spectral contrast
- **Emotions**: angry, disgust, fear, happy, neutral, pleasant_surprise, sad

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Use `app_simple.py` instead of `app.py`
   - Check that model files exist in the `model/` directory
   - Version compatibility issues with TensorFlow models

2. **Webcam Not Working**
   - Ensure your browser has camera permissions
   - Check if another application is using the camera
   - Try refreshing the page

3. **Audio Recording Issues**
   - Grant microphone permissions in your browser
   - Check system microphone settings
   - Ensure you're using HTTPS in production

4. **Port Already in Use**
   ```bash
   # Kill any existing processes
   netstat -ano | findstr :5000
   taskkill /PID <process_id> /F
   ```

### Model Compatibility
The current setup runs with:
- ✅ Text emotion detection (fully functional)
- ⚠️ Face emotion detection (demo mode with random predictions)
- ⚠️ Speech emotion detection (demo mode with random predictions)

To enable full functionality, you may need to:
- Retrain models with current TensorFlow/Keras versions
- Update model architectures for compatibility
- Use model conversion tools

## 💻 Technology Stack

### Backend
- **Flask** - Web framework
- **TensorFlow** - Deep learning models
- **scikit-learn** - Machine learning pipeline
- **librosa** - Audio processing
- **OpenCV** - Computer vision
- **SQLite** - Database storage

### Frontend
- **HTML5** - Structure and webcam/audio APIs
- **CSS3** - Responsive styling and animations
- **JavaScript** - Dynamic interactions and API calls
- **Chart.js** - Data visualization

### Libraries
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **neattext** - Text preprocessing
- **joblib** - Model serialization
- **Pillow** - Image processing

## 🚧 Development

### Adding New Models
1. Save your trained model in the `model/` directory
2. Update the model loading functions in `app.py`
3. Add appropriate preprocessing functions
4. Update emotion labels arrays
5. Test the integration

### Customizing the UI
- Modify `templates/index.html` for structure
- Update `static/css/style.css` for styling
- Enhance `static/js/app.js` for functionality

### Database Schema
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_type TEXT,
    predicted_emotion TEXT,
    confidence REAL,
    input_data TEXT
);
```

## Development

### Adding New Models
1. Save your trained model in the `model/` directory
2. Update the model loading functions in `app.py`
3. Add appropriate preprocessing functions
4. Update emotion labels arrays
5. Test the integration

### Customizing the UI
- Modify `templates/index.html` for structure
- Update `static/css/style.css` for styling
- Enhance `static/js/app.js` for functionality

### Database Schema
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_type TEXT,
    predicted_emotion TEXT,
    confidence REAL,
    input_data TEXT
);
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

This project is for educational and demonstration purposes. Please ensure you have the rights to use the pre-trained models in your specific use case.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- scikit-learn contributors for machine learning tools
- OpenCV community for computer vision capabilities
- librosa developers for audio processing features

## 👤 Author

**Balu Sayabanna Nandiwale**

- GitHub: [@BaluSayabannaNandiwale](https://github.com/BaluSayabannaNandiwale)
- LinkedIn: [Balu Sayabanna Nandiwale](https://www.linkedin.com/in/balusayabannandiwale/)

---

**Built with ❤️ for intelligent emotion detection**