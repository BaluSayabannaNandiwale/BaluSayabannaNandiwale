// Moodify - Intelligent Mood Detection System
// Main JavaScript file for handling UI interactions and API calls

class MoodifyApp {
    constructor() {
        this.webcamStream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.currentSection = 'face';
        this.emotionChart = null;
        
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupFileHandlers();
        this.setupWebcam();
        this.setupAudioRecording();
        this.setupTextInput();
        this.setupResultHandlers();
        this.setupToasts();
    }

    // Navigation handling
    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const sections = document.querySelectorAll('.detection-section');

        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                if (item.getAttribute('href').startsWith('/')) return; // Skip external links
                
                e.preventDefault();
                const targetSection = item.dataset.section;
                
                // Update active nav item
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                // Update active section
                sections.forEach(section => section.classList.remove('active'));
                document.getElementById(`${targetSection}-section`).classList.add('active');
                
                this.currentSection = targetSection;
                this.hideResults();
                this.resetCurrentSection();
            });
        });
    }

    // File upload handlers
    setupFileHandlers() {
        // Face image upload
        const faceUploadZone = document.getElementById('face-upload-zone');
        const faceFileInput = document.getElementById('face-file-input');
        
        faceUploadZone.addEventListener('click', () => faceFileInput.click());
        faceUploadZone.addEventListener('dragover', this.handleDragOver);
        faceUploadZone.addEventListener('drop', (e) => this.handleImageDrop(e, faceFileInput));
        faceFileInput.addEventListener('change', (e) => this.handleImageSelect(e));

        // Speech audio upload
        const speechUploadZone = document.getElementById('speech-upload-zone');
        const speechFileInput = document.getElementById('speech-file-input');
        
        speechUploadZone.addEventListener('click', () => speechFileInput.click());
        speechUploadZone.addEventListener('dragover', this.handleDragOver);
        speechUploadZone.addEventListener('drop', (e) => this.handleAudioDrop(e, speechFileInput));
        speechFileInput.addEventListener('change', (e) => this.handleAudioSelect(e));
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.style.background = '#e9ecef';
    }

    handleImageDrop(e, fileInput) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.style.background = '#f8f9fa';
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            fileInput.files = files;
            this.handleImageSelect({ target: fileInput });
        } else {
            this.showError('Please drop a valid image file.');
        }
    }

    handleAudioDrop(e, fileInput) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.style.background = '#f8f9fa';
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('audio/')) {
            fileInput.files = files;
            this.handleAudioSelect({ target: fileInput });
        } else {
            this.showError('Please drop a valid audio file.');
        }
    }

    handleImageSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        console.log('Image selected:', file.name, file.type);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('face-preview');
            const previewImg = document.getElementById('face-preview-img');
            
            previewImg.src = e.target.result;
            preview.style.display = 'block';
            
            // Setup analyze button
            const analyzeBtn = document.getElementById('analyze-face');
            analyzeBtn.onclick = () => {
                console.log('Analyze button clicked for image');
                this.analyzeImage(file);
            };
        };
        reader.readAsDataURL(file);
    }

    handleAudioSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        console.log('Audio selected:', file.name, file.type, file.size);
        
        // Validate file size (limit to 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Audio file is too large. Please select a file smaller than 10MB.');
            return;
        }
        
        const preview = document.getElementById('speech-preview');
        const previewAudio = document.getElementById('speech-preview-audio');
        
        const url = URL.createObjectURL(file);
        previewAudio.src = url;
        preview.style.display = 'block';
        
        // Setup analyze button
        const analyzeBtn = document.getElementById('analyze-speech');
        analyzeBtn.onclick = () => {
            console.log('Analyze button clicked for audio');
            this.analyzeAudio(file);
        };
    }

    // Webcam functionality
    setupWebcam() {
        const startBtn = document.getElementById('start-webcam');
        const captureBtn = document.getElementById('capture-photo');
        const stopBtn = document.getElementById('stop-webcam');
        const webcam = document.getElementById('webcam');

        startBtn.addEventListener('click', () => this.startWebcam());
        captureBtn.addEventListener('click', () => this.capturePhoto());
        stopBtn.addEventListener('click', () => this.stopWebcam());
    }

    async startWebcam() {
        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const webcam = document.getElementById('webcam');
            webcam.srcObject = this.webcamStream;
            
            document.getElementById('start-webcam').disabled = true;
            document.getElementById('capture-photo').disabled = false;
            document.getElementById('stop-webcam').disabled = false;
            
        } catch (error) {
            this.showError('Error accessing webcam: ' + error.message);
        }
    }

    capturePhoto() {
        const webcam = document.getElementById('webcam');
        const canvas = document.getElementById('webcam-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        ctx.drawImage(webcam, 0, 0);
        
        // Convert to blob and show preview
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const preview = document.getElementById('face-preview');
            const previewImg = document.getElementById('face-preview-img');
            
            previewImg.src = url;
            preview.style.display = 'block';
            
            // Setup analyze button with captured image data
            const analyzeBtn = document.getElementById('analyze-face');
            analyzeBtn.onclick = () => this.analyzeWebcamImage(canvas);
        });
    }

    stopWebcam() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        
        document.getElementById('start-webcam').disabled = false;
        document.getElementById('capture-photo').disabled = true;
        document.getElementById('stop-webcam').disabled = true;
        
        const webcam = document.getElementById('webcam');
        webcam.srcObject = null;
    }

    // Audio recording functionality
    setupAudioRecording() {
        const startBtn = document.getElementById('start-recording');
        const stopBtn = document.getElementById('stop-recording');
        const playBtn = document.getElementById('play-recording');

        startBtn.addEventListener('click', () => this.startRecording());
        stopBtn.addEventListener('click', () => this.stopRecording());
        playBtn.addEventListener('click', () => this.playRecording());
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.recordedChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, { type: 'audio/wav' });
                const url = URL.createObjectURL(blob);
                
                const recordedAudio = document.getElementById('recorded-audio');
                recordedAudio.src = url;
                recordedAudio.style.display = 'block';
                
                document.getElementById('play-recording').disabled = false;
                
                // Setup analyze button
                const preview = document.getElementById('speech-preview');
                const previewAudio = document.getElementById('speech-preview-audio');
                previewAudio.src = url;
                preview.style.display = 'block';
                
                const analyzeBtn = document.getElementById('analyze-speech');
                analyzeBtn.onclick = () => this.analyzeRecordedAudio(blob);
            };
            
            this.mediaRecorder.start(100);
            this.startAudioVisualization(stream);
            
            document.getElementById('start-recording').disabled = true;
            document.getElementById('stop-recording').disabled = false;
            
        } catch (error) {
            this.showError('Error accessing microphone: ' + error.message);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        document.getElementById('start-recording').disabled = false;
        document.getElementById('stop-recording').disabled = true;
    }

    playRecording() {
        const audio = document.getElementById('recorded-audio');
        audio.play();
    }

    startAudioVisualization(stream) {
        const canvas = document.getElementById('audio-canvas');
        const ctx = canvas.getContext('2d');
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        
        source.connect(analyser);
        analyser.fftSize = 256;
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                requestAnimationFrame(draw);
                
                analyser.getByteFrequencyData(dataArray);
                
                ctx.fillStyle = '#f8f9fa';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let barHeight;
                let x = 0;
                
                for (let i = 0; i < bufferLength; i++) {
                    barHeight = dataArray[i] / 255 * canvas.height;
                    
                    ctx.fillStyle = `hsl(${i / bufferLength * 360}, 50%, 50%)`;
                    ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                    
                    x += barWidth + 1;
                }
            }
        };
        
        draw();
    }

    // Text input functionality
    setupTextInput() {
        const textInput = document.getElementById('text-input');
        const charCount = document.querySelector('.char-count');
        const clearBtn = document.getElementById('clear-text');
        const analyzeBtn = document.getElementById('analyze-text');

        textInput.addEventListener('input', () => {
            const count = textInput.value.length;
            charCount.textContent = `${count}/1000`;
            
            if (count > 800) {
                charCount.style.color = '#dc3545';
            } else if (count > 600) {
                charCount.style.color = '#fd7e14';
            } else {
                charCount.style.color = '#666';
            }
        });

        clearBtn.addEventListener('click', () => {
            textInput.value = '';
            charCount.textContent = '0/1000';
            charCount.style.color = '#666';
        });

        analyzeBtn.addEventListener('click', () => {
            const text = textInput.value.trim();
            if (text) {
                this.analyzeText(text);
            } else {
                this.showError('Please enter some text to analyze.');
            }
        });
        
        // Add test demo button functionality
        document.addEventListener('click', (e) => {
            if (e.target.id === 'test-demo' || e.target.closest('#test-demo')) {
                console.log('Test demo button clicked');
                this.testDemo();
            }
        });
    }

    // Analysis functions
    async analyzeImage(file) {
        try {
            this.showLoading();
            
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch('/predict_face', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result, 'face');
                this.showSuccess('Face emotion analysis completed!');
            } else {
                this.showError(result.error || 'Error analyzing image');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
            console.error('Analysis error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeWebcamImage(canvas) {
        try {
            this.showLoading();
            
            canvas.toBlob(async (blob) => {
                try {
                    const formData = new FormData();
                    formData.append('image', blob, 'webcam-capture.jpg');
                    
                    const response = await fetch('/predict_face', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        this.displayResults(result, 'face');
                        this.showSuccess('Face emotion analysis completed!');
                    } else {
                        this.showError(result.error || 'Error analyzing image');
                    }
                } catch (error) {
                    this.showError('Network error: ' + error.message);
                    console.error('Analysis error:', error);
                } finally {
                    this.hideLoading();
                }
            }, 'image/jpeg', 0.8);
        } catch (error) {
            this.showError('Error processing webcam image: ' + error.message);
            this.hideLoading();
        }
    }

    async analyzeAudio(file) {
        try {
            this.showLoading();
            
            const formData = new FormData();
            formData.append('audio', file);
            
            const response = await fetch('/predict_speech', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result, 'speech');
                this.showSuccess('Speech emotion analysis completed!');
            } else {
                this.showError(result.error || 'Error analyzing audio');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
            console.error('Analysis error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeRecordedAudio(blob) {
        try {
            this.showLoading();
            
            const formData = new FormData();
            formData.append('audio', blob, 'recorded-audio.wav');
            
            const response = await fetch('/predict_speech', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result, 'speech');
                this.showSuccess('Speech emotion analysis completed!');
            } else {
                this.showError(result.error || 'Error analyzing audio');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
            console.error('Analysis error:', error);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeText(text) {
        try {
            this.showLoading();
            
            const response = await fetch('/predict_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result, 'text');
                this.showSuccess('Text emotion analysis completed!');
            } else {
                this.showError(result.error || 'Error analyzing text');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
            console.error('Analysis error:', error);
        } finally {
            this.hideLoading();
        }
    }

    // Results display
    displayResults(result, type) {
        // Remember last detected emotion for chat handoff
        this.lastDetectedEmotion = result.emotion;
        const resultsSection = document.getElementById('results-section');
        const emotionIcon = document.getElementById('emotion-icon');
        const detectedEmotion = document.getElementById('detected-emotion');
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceFill = document.getElementById('confidence-fill');
        
        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Set emotion icon and text
        const emotionIcons = {
            'happy': '😊', 'joy': '😊',
            'sad': '😢', 'sadness': '😢',
            'angry': '😠', 'anger': '😠',
            'fear': '😨',
            'surprise': '😮',
            'disgust': '🤢',
            'neutral': '😐',
            'pleasant_surprise': '😊',
            'shame': '😳'
        };
        
        const emotion = result.emotion.toLowerCase();
        emotionIcon.textContent = emotionIcons[emotion] || '😐';
        detectedEmotion.textContent = result.emotion.charAt(0).toUpperCase() + result.emotion.slice(1);
        
        // Set confidence
        const confidence = Math.round(result.confidence * 100);
        confidenceValue.textContent = `${confidence}%`;
        confidenceFill.style.width = `${confidence}%`;
        
        // Wire Talk to AI button
        const talkBtn = document.getElementById('talk-to-ai');
        if (talkBtn) {
            const emotionParam = encodeURIComponent(this.lastDetectedEmotion || 'neutral');
            talkBtn.href = `/chat?emotion=${emotionParam}`;
        }

        // Create emotion chart
        this.createEmotionChart(result.probabilities);
    }

    createEmotionChart(probabilities) {
        const canvas = document.getElementById('emotion-chart');
        // Fix canvas dimensions for consistent sizing
        canvas.width = 700;
        canvas.height = 280;
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.emotionChart) {
            this.emotionChart.destroy();
        }
        
        const emotions = Object.keys(probabilities);
        const values = Object.values(probabilities).map(v => Math.round(v * 100));
        
        this.emotionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                datasets: [{
                    label: 'Confidence (%)',
                    data: values,
                    backgroundColor: [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
                    ].slice(0, emotions.length),
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2.5,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            },
                            color: 'rgba(255, 255, 255, 0.8)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.8)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        }
                    }
                }
            }
        });
    }

    // Result handlers
    setupResultHandlers() {
        const tryAgainBtn = document.getElementById('try-again');
        const saveResultBtn = document.getElementById('save-result');
        
        tryAgainBtn.addEventListener('click', () => {
            this.hideResults();
            this.resetCurrentSection();
        });
        
        saveResultBtn.addEventListener('click', () => {
            this.showSuccess('Result saved to your history!');
        });
    }

    // Utility functions
    hideResults() {
        document.getElementById('results-section').style.display = 'none';
    }

    resetCurrentSection() {
        // Reset face section
        document.getElementById('face-preview').style.display = 'none';
        document.getElementById('face-file-input').value = '';
        
        // Reset speech section
        document.getElementById('speech-preview').style.display = 'none';
        document.getElementById('speech-file-input').value = '';
        document.getElementById('recorded-audio').style.display = 'none';
        
        // Reset text section
        document.getElementById('text-input').value = '';
        document.querySelector('.char-count').textContent = '0/1000';
        
        // Stop webcam if running
        this.stopWebcam();
        
        // Stop recording if active
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.stopRecording();
        }
    }

    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    // Toast notifications
    setupToasts() {
        // Setup close buttons for toasts
        document.querySelectorAll('.toast-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.toast').style.display = 'none';
            });
        });
    }

    showError(message) {
        const toast = document.getElementById('error-toast');
        const messageElement = document.getElementById('error-message');
        
        if (toast && messageElement) {
            messageElement.textContent = message;
            toast.style.display = 'flex';
            
            setTimeout(() => {
                toast.style.display = 'none';
            }, 5000);
        } else {
            // Fallback to alert if toast elements don't exist
            alert('Error: ' + message);
        }
    }

    showSuccess(message) {
        const toast = document.getElementById('success-toast');
        const messageElement = document.getElementById('success-message');
        
        if (toast && messageElement) {
            messageElement.textContent = message;
            toast.style.display = 'flex';
            
            setTimeout(() => {
                toast.style.display = 'none';
            }, 3000);
        } else {
            // Fallback to alert if toast elements don't exist
            console.log('Success: ' + message);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MoodifyApp();
});

// Test demo function for debugging
function testDemo() {
    console.log('Running test demo...');
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
    
    // Simulate API call with timeout
    setTimeout(() => {
        const demoResult = {
            emotion: 'happy',
            confidence: 0.85,
            probabilities: {
                'happy': 0.85,
                'neutral': 0.10,
                'surprise': 0.03,
                'sad': 0.01,
                'angry': 0.01,
                'fear': 0.00,
                'disgust': 0.00
            }
        };
        
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        // Show results
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        console.log('Demo completed successfully');
    }, 2000);
}