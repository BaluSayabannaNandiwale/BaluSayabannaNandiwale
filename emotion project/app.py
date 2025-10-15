from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import cv2
import numpy as np
import pandas as pd
import io
import base64
from PIL import Image
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
import hashlib
import secrets

# Note: heavy ML libraries (tensorflow, librosa, joblib, neattext, google.genai) are imported
# lazily inside the functions that need them. This avoids long imports during Flask's
# reloader restarts which can cause the watchdog to hang on Windows.
import io
import base64
from PIL import Image
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
import hashlib
import secrets
import io
import base64
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = secrets.token_hex(16)  # Generate a random secret key for sessions

# Load environment variables from .env if present
load_dotenv()

# Global variables for models
face_model = None
speech_model = None
text_model = None
text_model_loading = False
text_model_load_error = None

# Emotion labels
FACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SPEECH_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
TEXT_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']

def load_models():
    """Load all pre-trained models"""
    global face_model, speech_model, text_model
    
    try:
        # Start loading text emotion model in background to avoid blocking
        try:
            import threading
            def _load_text_model_background():
                global text_model, text_model_loading, text_model_load_error
                try:
                    text_model_loading = True
                    import joblib
                    text_model = joblib.load('model/text_emotion.pkl')
                    print("Text emotion model loaded successfully (background)")
                    text_model_load_error = None
                except Exception as e:
                    text_model = None
                    text_model_load_error = str(e)
                    print(f"Text model loading failed (background): {e}")
                finally:
                    text_model_loading = False

            t = threading.Thread(target=_load_text_model_background, daemon=True)
            t.start()
        except Exception as e:
            print(f"Failed to start background text model loader: {e}")
        
        # Try loading face emotion model
        try:
            face_model_path = os.path.join('model', 'facialemotionmodel.h5')
            if os.path.exists(face_model_path):
                # import tensorflow locally because it can be slow and may interfere with
                # Flask's reloader on Windows if imported at module level
                import tensorflow as tf
                face_model = tf.keras.models.load_model(face_model_path, compile=False)
                print("Face emotion model loaded successfully")
            else:
                face_model = None
                print(f"Face model file not found at {face_model_path}")
        except Exception as e:
            face_model = None
            print(f"Face model loading failed: {e}")
        
        # Try loading speech emotion model
        try:
            speech_model_path = os.path.join('model', 'speech_emotion_model.h5')
            if os.path.exists(speech_model_path):
                # tensorflow already imported above if face model loaded; import again locally
                import tensorflow as tf
                speech_model = tf.keras.models.load_model(speech_model_path, compile=False)
                print("Speech emotion model loaded successfully")
            else:
                speech_model = None
                print(f"Speech model file not found at {speech_model_path}")
        except Exception as e:
            speech_model = None
            print(f"Speech model loading failed: {e}")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")

def preprocess_face_image(image_data):
    """Preprocess image for face emotion detection"""
    try:
        # Convert base64 to image
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image.convert('RGB'))
        else:
            # Ensure numpy array in RGB
            image = image_data
            if image is None:
                return None, "Invalid image data"
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Resize very large images for better detection performance
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
            h, w = image.shape[:2]
        
        # Convert to grayscale and normalize contrast
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect face using multiple cascades and parameter sweeps
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_alt_tree.xml'
        ]
        faces = []
        for cf in cascade_files:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cf)
            if cascade.empty():
                continue
            for scale_factor in [1.1, 1.2, 1.3]:
                for neighbors in [3, 5, 7]:
                    detected = cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=neighbors,
                        minSize=(48, 48)
                    )
                    if len(detected) > 0:
                        faces = detected
                        break
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break
        
        # Fallback: use center crop if no faces found
        if len(faces) == 0:
            side = int(min(h, w) * 0.6)
            cx, cy = w // 2, h // 2
            x = max(0, cx - side // 2)
            y = max(0, cy - side // 2)
            faces = [(x, y, side, side)]
        
        # Use the first detected face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to 48x48 for the model
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values
        face_normalized = face_resized / 255.0
        
        # Reshape for model input
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        return face_input, None
        
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def extract_mfcc_features(audio_data, sample_rate=22050):
    """Extract MFCC features from audio for speech emotion detection"""
    try:
        # import librosa locally to avoid importing it at module load time
        import librosa
        # Guard: ensure non-empty audio
        if audio_data is None or len(audio_data) == 0:
            return None

        # Trim leading/trailing silence and normalize
        audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=30)
        if len(audio_trimmed) == 0:
            audio_trimmed = audio_data
        audio = librosa.util.normalize(audio_trimmed) if np.max(np.abs(audio_trimmed)) > 0 else audio_trimmed

        # Ensure minimum duration by padding (0.5 sec)
        min_len = int(0.5 * sample_rate)
        if len(audio) < min_len:
            pad = np.zeros(min_len - len(audio))
            audio = np.concatenate([audio, pad])

        # Extract robust set of features with fallbacks
        feats = []
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            feats.append(np.mean(mfccs.T, axis=0))
        except Exception:
            return None
        
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            feats.append(np.mean(chroma.T, axis=0))
        except Exception:
            pass

        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            feats.append(np.mean(mel.T, axis=0))
        except Exception:
            pass

        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            feats.append(np.mean(contrast.T, axis=0))
        except Exception:
            pass

        try:
            harm = librosa.effects.harmonic(audio)
            tonnetz = librosa.feature.tonnetz(y=harm, sr=sample_rate)
            feats.append(np.mean(tonnetz.T, axis=0))
        except Exception:
            pass

        if not feats:
            return None

        features = np.hstack(feats)
        return features.reshape(1, -1)
     
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def preprocess_text(text):
    """Preprocess text for emotion detection"""
    try:
        # Import neattext functions lazily
        try:
            import neattext.functions as nfx
            # Remove user handles and stopwords
            clean_text = nfx.remove_userhandles(text)
            clean_text = nfx.remove_stopwords(clean_text)
        except Exception:
            # If neattext isn't available, fall back to a simple cleanup
            clean_text = ' '.join(word for word in text.split() if not word.startswith('@'))
        return clean_text
    except Exception as e:
        return text

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Initialize SQLite database for storing predictions and users"""
    conn = sqlite3.connect('emotion_history.db')
    cursor = conn.cursor()
    
    # Create users table first
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_type TEXT,
            predicted_emotion TEXT,
            confidence REAL,
            input_data TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Check if user_id column exists in predictions table, if not add it
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'user_id' not in columns:
        print("Adding user_id column to predictions table...")
        cursor.execute('ALTER TABLE predictions ADD COLUMN user_id INTEGER')
    
    conn.commit()
    conn.close()

def get_user_by_username(username):
    """Get user by username"""
    conn = sqlite3.connect('emotion_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_email(email):
    """Get user by email"""
    conn = sqlite3.connect('emotion_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, email, password):
    """Create a new user"""
    conn = sqlite3.connect('emotion_history.db')
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute('''
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?)
    ''', (username, email, password_hash))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id

def update_last_login(user_id):
    """Update user's last login time"""
    conn = sqlite3.connect('emotion_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
    ''', (user_id,))
    conn.commit()
    conn.close()

def login_required(f):
    """Decorator to require login for certain routes"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def save_prediction(input_type, emotion, confidence, input_data=""):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('emotion_history.db')
        cursor = conn.cursor()
        
        user_id = session.get('user_id')  # Get user_id from session, None if not logged in
        
        cursor.execute('''
            INSERT INTO predictions (input_type, predicted_emotion, confidence, input_data, user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (input_type, emotion, confidence, input_data[:100], user_id))  # Limit input_data length
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html', session=session)

@app.route('/app')
def app_page():
    """Main application page"""
    return render_template('index.html', session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')
        
        user = get_user_by_username(username)
        if user and user[3] == hash_password(password):  # user[3] is password_hash
            session['user_id'] = user[0]  # user[0] is id
            session['username'] = user[1]  # user[1] is username
            update_last_login(user[0])
            flash('Login successful!', 'success')
            return redirect(url_for('app_page'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
        
        # Check if username already exists
        if get_user_by_username(username):
            flash('Username already exists.', 'error')
            return render_template('signup.html')
        
        # Check if email already exists
        if get_user_by_email(email):
            flash('Email already exists.', 'error')
            return render_template('signup.html')
        
        try:
            user_id = create_user(username, email, password)
            session['user_id'] = user_id
            session['username'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('app_page'))
        except Exception as e:
            flash('Error creating account. Please try again.', 'error')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/predict_face', methods=['POST'])
def predict_face():
    """Predict emotion from face image"""
    try:
        if face_model is None:
            return jsonify({'error': 'Face emotion model not available on server'}), 500
            
        if 'image' not in request.files and 'imageData' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
        else:
            # Base64 image data from webcam
            image_data = request.json.get('imageData', '')
            image_array = image_data
        
        # Preprocess image
        processed_image, error = preprocess_face_image(image_array)
        if error:
            return jsonify({'error': error}), 400
        
        # Make prediction
        prediction = face_model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        emotion = FACE_EMOTIONS[predicted_class]
        
        # Get all probabilities
        probabilities = {FACE_EMOTIONS[i]: float(prediction[0][i]) for i in range(len(FACE_EMOTIONS))}
        
        # Save to database
        save_prediction('face', emotion, confidence)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    """Predict emotion from speech audio"""
    try:
        # Import librosa locally (we use lazy imports to avoid heavy startup costs)
        import librosa
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load audio file
        audio_bytes = file.read()
        
        # Save temporarily to process with librosa
        temp_path = 'temp_audio.wav'
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(audio_bytes)
        
        # Load audio with librosa (mono, target sr)
        try:
            audio_data, sample_rate = librosa.load(temp_path, sr=22050, mono=True)
        except Exception:
            # Fallback: try without resample
            audio_data, sample_rate = librosa.load(temp_path, mono=True)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Extract features
        features = extract_mfcc_features(audio_data, sample_rate)
        if features is None:
            return jsonify({'error': 'Error extracting audio features'}), 400
        
        # Make prediction (model or heuristic)
        if speech_model is not None:
            prediction = speech_model.predict(features)
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            emotion = SPEECH_EMOTIONS[predicted_class]
            probabilities = {SPEECH_EMOTIONS[i]: float(prediction[0][i]) for i in range(len(SPEECH_EMOTIONS))}
        else:
            # Heuristic fallback
            try:
                rms = float(np.mean(librosa.feature.rms(y=audio_data)))
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio_data)))
                centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)))
            except Exception:
                rms, zcr, centroid = 0.05, 0.05, 1000.0
            # Normalize crude ranges
            def clamp01(x):
                return max(0.0, min(1.0, x))
            arousal = clamp01((rms - 0.02) / 0.1)  # ~0-0.12 typical
            brightness = clamp01((centroid - 800) / 2500)
            noisiness = clamp01(zcr / 0.2)
            scores = {
                'happy': 0.4*arousal + 0.4*brightness + 0.2*(1-noisiness),
                'sad': 0.7*(1-arousal) + 0.2*(1-brightness) + 0.1*(1-noisiness),
                'angry': 0.5*arousal + 0.3*noisiness + 0.2*brightness,
                'neutral': 1.0 - abs(arousal-0.5) - 0.2*abs(brightness-0.5),
                'fear': 0.5*noisiness + 0.3*arousal + 0.2*(1-brightness),
                'disgust': 0.5*noisiness + 0.3*(1-brightness) + 0.2*arousal,
                'pleasant_surprise': 0.6*brightness + 0.3*arousal + 0.1*(1-noisiness)
            }
            # Softmax over the available label set
            logits = np.array([scores[l] for l in SPEECH_EMOTIONS])
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            predicted_class = int(np.argmax(probs))
            emotion = SPEECH_EMOTIONS[predicted_class]
            confidence = float(probs[predicted_class])
            probabilities = {SPEECH_EMOTIONS[i]: float(probs[i]) for i in range(len(SPEECH_EMOTIONS))}
        
        # Save to database
        save_prediction('speech', emotion, confidence)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Predict emotion from text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess text
        clean_text = preprocess_text(text)

        # If the text model is currently loading, inform the client
        global text_model_loading, text_model_load_error, text_model
        if text_model_loading:
            return jsonify({'error': 'Text model is still loading - try again shortly'}), 503

        # If the model failed to load, attempt on-demand load (synchronously) once
        if text_model is None:
            if text_model_load_error:
                # Try a single synchronous load attempt to surface a useful error
                try:
                    import joblib
                    text_model = joblib.load('model/text_emotion.pkl')
                    text_model_load_error = None
                    print('Text model loaded on-demand')
                except Exception as e:
                    text_model_load_error = str(e)
                    return jsonify({'error': 'Text model not available', 'details': text_model_load_error}), 503
            else:
                # No prior error but model is not loaded; attempt load
                try:
                    import joblib
                    text_model = joblib.load('model/text_emotion.pkl')
                except Exception as e:
                    text_model_load_error = str(e)
                    return jsonify({'error': 'Text model not available', 'details': text_model_load_error}), 503

        # Make prediction
        prediction = text_model.predict([clean_text])
        emotion = prediction[0]

        # Get prediction probabilities
        try:
            prediction_proba = text_model.predict_proba([clean_text])
            confidence = float(np.max(prediction_proba[0]))

            # Get all probabilities
            probabilities = {emotion_class: float(prob)
                           for emotion_class, prob in zip(text_model.classes_, prediction_proba[0])}
        except Exception:
            confidence = 1.0
            probabilities = {emotion: 1.0}
        
        # Save to database
        save_prediction('text', emotion, confidence, text[:50])
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing text: {str(e)}'}), 500

@app.route('/analytics')
def analytics():
    """Show analytics dashboard"""
    try:
        conn = sqlite3.connect('emotion_history.db')
        
        # Check if user_id column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'user_id' in columns and 'user_id' in session:
            # Get user-specific predictions
            df = pd.read_sql_query("""
                SELECT * FROM predictions 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT 100
            """, conn, params=(session['user_id'],))
        else:
            # Get all predictions (for backward compatibility or non-logged users)
            df = pd.read_sql_query("""
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 100
            """, conn)
        
        conn.close()
        
        if len(df) == 0:
            return render_template('analytics.html', 
                                 recent_predictions=[],
                                 emotion_counts={},
                                 avg_confidence=0,
                                 session=session)
        
        # Calculate statistics
        emotion_counts = df['predicted_emotion'].value_counts().to_dict()
        avg_confidence = df['confidence'].mean()
        
        # Recent predictions
        recent = df.head(10).to_dict('records')
        
        return render_template('analytics.html',
                             recent_predictions=recent,
                             emotion_counts=emotion_counts,
                             avg_confidence=avg_confidence,
                             session=session)
        
    except Exception as e:
        return jsonify({'error': f'Error loading analytics: {str(e)}'}), 500

@app.route('/chat')
def chat_page():
    return render_template('chat.html', session=session)


@app.route('/status')
def status():
    """Return current model load status and basic info"""
    try:
        status_info = {
            'face_model_loaded': bool(face_model),
            'speech_model_loaded': bool(speech_model),
            'text_model_loaded': bool(text_model),
            'text_model_loading': bool(text_model_loading),
            'text_model_load_error': text_model_load_error
        }
        return jsonify(status_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        # Import Google GenAI SDK lazily to avoid heavy imports during module import
        try:
            import google.generativeai as genai
        except Exception:
            genai = None
        try:
            from google import genai as genai_new
        except Exception:
            genai_new = None
        data = request.get_json() or {}
        prompt = data.get('prompt', '').strip()
        emotion = (data.get('emotion') or 'neutral').strip().lower()
        meta = data.get('meta') or {}

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Fallback responses when AI service is not available
        def get_fallback_response(prompt, emotion):
            prompt_lower = prompt.lower()
            
            # Music suggestions
            if 'music' in prompt_lower:
                music_suggestions = {
                    'happy': "For your happy mood, try upbeat songs like 'Happy' by Pharrell Williams, 'Good Vibrations' by The Beach Boys, or 'Walking on Sunshine' by Katrina and the Waves! 🎵",
                    'sad': "For your sad mood, try comforting songs like 'Lean on Me' by Bill Withers, 'You've Got a Friend' by James Taylor, or 'Bridge Over Troubled Water' by Simon & Garfunkel. 💙",
                    'angry': "For your angry mood, try energetic songs like 'Eye of the Tiger' by Survivor, 'We Will Rock You' by Queen, or 'Stronger' by Kanye West to channel that energy! 💪",
                    'fear': "For your fearful mood, try calming songs like 'Here Comes the Sun' by The Beatles, 'Three Little Birds' by Bob Marley, or 'Don't Worry Be Happy' by Bobby McFerrin. 🌟",
                    'surprise': "For your surprised mood, try exciting songs like 'Thriller' by Michael Jackson, 'Bohemian Rhapsody' by Queen, or 'Sweet Child O' Mine' by Guns N' Roses! ✨",
                    'disgust': "For your frustrated mood, try empowering songs like 'I Will Survive' by Gloria Gaynor, 'Fight Song' by Rachel Platten, or 'Roar' by Katy Perry! 🦁",
                    'neutral': "For your neutral mood, try balanced songs like 'Imagine' by John Lennon, 'What a Wonderful World' by Louis Armstrong, or 'Hallelujah' by Leonard Cohen. 🎶"
                }
                return music_suggestions.get(emotion, music_suggestions['neutral'])
            
            # Movie suggestions
            elif 'movie' in prompt_lower:
                movie_suggestions = {
                    'happy': "For your happy mood, try uplifting movies like 'The Pursuit of Happyness', 'Up', 'The Secret Life of Walter Mitty', or 'La La Land'! 🎬",
                    'sad': "For your sad mood, try heartwarming movies like 'The Shawshank Redemption', 'Forrest Gump', 'The Lion King', or 'Inside Out' to help process emotions. 💙",
                    'angry': "For your angry mood, try action movies like 'Mad Max: Fury Road', 'John Wick', 'The Dark Knight', or 'Gladiator' to channel that energy! 💪",
                    'fear': "For your fearful mood, try inspiring movies like 'The Wizard of Oz', 'Finding Nemo', 'The Princess Bride', or 'Spirited Away' for comfort. 🌟",
                    'surprise': "For your surprised mood, try thrilling movies like 'Inception', 'The Matrix', 'Interstellar', or 'Avatar' for excitement! ✨",
                    'disgust': "For your frustrated mood, try empowering movies like 'Rocky', 'The Pursuit of Happyness', 'Hidden Figures', or 'Wonder Woman'! 🦸",
                    'neutral': "For your neutral mood, try classic movies like 'Casablanca', 'The Godfather', 'Citizen Kane', or 'Schindler's List'. 🎭"
                }
                return movie_suggestions.get(emotion, movie_suggestions['neutral'])
            
            # Support suggestions
            elif 'support' in prompt_lower or 'help' in prompt_lower:
                return f"I'm here to support you! Based on your {emotion} mood, here are some resources:\n\n" + \
                       "• **Mental Health Hotlines:**\n" + \
                       "  - National Suicide Prevention Lifeline: 988\n" + \
                       "  - Crisis Text Line: Text HOME to 741741\n" + \
                       "  - SAMHSA National Helpline: 1-800-662-4357\n\n" + \
                       "• **Online Resources:**\n" + \
                       "  - BetterHelp.com for online therapy\n" + \
                       "  - Headspace.com for meditation\n" + \
                       "  - 7cups.com for peer support\n\n" + \
                       "• **Self-Care Tips:**\n" + \
                       "  - Take deep breaths\n" + \
                       "  - Go for a walk\n" + \
                       "  - Talk to a friend\n" + \
                       "  - Practice gratitude\n\n" + \
                       "Remember, it's okay to not be okay. You're not alone! 💙"
            
            # General responses
            else:
                general_responses = {
                    'happy': f"I'm so glad you're feeling happy! 😊 That's wonderful! Is there anything specific you'd like to talk about or any way I can help you celebrate this positive mood?",
                    'sad': f"I'm sorry you're feeling sad. 💙 It's completely normal to have these feelings. Would you like to talk about what's on your mind, or would you prefer some suggestions for things that might help?",
                    'angry': f"I can sense you're feeling angry. 💪 That's a valid emotion. Would you like to talk about what's bothering you, or would you prefer some strategies to help manage these feelings?",
                    'fear': f"I understand you might be feeling fearful or anxious. 🌟 That can be really challenging. Would you like to talk about what's worrying you, or would you prefer some calming techniques?",
                    'surprise': f"Something surprising happened! ✨ That can be exciting or overwhelming. Would you like to share what surprised you, or is there anything I can help you process?",
                    'disgust': f"I can sense you're feeling frustrated or disgusted. 🤝 That's completely understandable. Would you like to talk about what's bothering you, or would you prefer some ways to work through these feelings?",
                    'neutral': f"Hi there! 💬 I'm here to chat and help however I can. What's on your mind today? I'm ready to listen and support you!"
                }
                return general_responses.get(emotion, general_responses['neutral'])

        # Try to use Google AI if available
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            try:
                system = (
                    "You are Moodify Assistant, an empathetic, concise AI that adapts to the user's detected emotion. "
                    "Offer helpful, safe suggestions for music, movies, and mental health resources when asked. "
                    "Tone should be supportive and non-judgmental. If the user asks for help, include reputable hotlines and online resources appropriate for general audiences. "
                    f"Detected emotion context: {emotion}."
                )

                # Prefer new Google genai client if available
                composed = f"System:\n{system}\n\nUser:\n{prompt}"
                if genai_new is not None:
                    try:
                        client = genai_new.Client(api_key=google_key)
                        # Try different models
                        for m in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
                            try:
                                response = client.models.generate_content(
                                    model=m,
                                    contents=composed
                                )
                                reply = (getattr(response, 'text', None) or '').strip()
                                if reply:
                                    return jsonify({"reply": reply})
                            except Exception as inner_e:
                                continue
                    except Exception as e:
                        pass
                
                # Fallback to legacy SDK
                genai.configure(api_key=google_key)
                candidate_models = ["gemini-1.5-flash", "gemini-1.5-pro"]

                for model_name in candidate_models:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(composed)
                        reply = (getattr(response, 'text', None) or '').strip()
                        if reply:
                            return jsonify({"reply": reply})
                    except Exception as e:
                        continue
            except Exception as e:
                pass  # Fall through to fallback response

        # Use fallback response if AI service fails
        reply = get_fallback_response(prompt, emotion)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Load models
    load_models()
    
    # Run the app
    # Disable the Flask reloader on Windows to avoid double-imports and TensorFlow
    # hang during the watchdog restart. The reloader can be turned on during
    # development if needed, but make sure heavy ML imports are lazy.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)