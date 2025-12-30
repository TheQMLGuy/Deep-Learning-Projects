
import sys
import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
import torch
import torch.nn as nn
from PIL import Image
import os
import time
import joblib

# Optional Imports (Handle Failures Gracefully)
try:
    from torchvision import models
    TORCHVISION_AVAIL = True
except:
    print("WARNING: torchvision not found. ResNet will be unavailable.")
    TORCHVISION_AVAIL = False

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAIL = True
except:
    print("WARNING: facenet-pytorch not found. Embedding model will be unavailable.")
    FACENET_AVAIL = False

try:
    from skimage.feature import hog
    from skimage.transform import resize
    SKIMAGE_AVAIL = True
except:
    print("WARNING: skimage not found. SVM will be unavailable.")
    SKIMAGE_AVAIL = False

# ==========================================
# 1. Model Definitions
# ==========================================

# --- Custom CNN ---
class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # 100x100 -> 50x50

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool -> 50x50 -> 25x25

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool -> 25x25 -> 12x12

        self.flatten = nn.Flatten()
        
        # Fully Connected Layers
        # 12*12*128 = 18432
        self.fc1 = nn.Linear(12 * 12 * 128, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2) # 2 Classes: Male, Female

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x) # No dropout during inference
        x = self.fc2(x)
        return x

# --- Embedding Classifier ---
class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingClassifier, self).__init__()
        self.resnet = InceptionResnetV1(pretrained=None) # Start blank, load weights later
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# ==========================================
# 2. Strategy Pattern for Models
# ==========================================
class ModelStrategy:
    def predict(self, face_pil):
        pass

class CNNStrategy(ModelStrategy):
    def __init__(self, path, device):
        self.device = device
        self.model = GenderCNN().to(device)
        self.loaded = False
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, map_location=device))
                self.model.eval()
                self.loaded = True
            except Exception as e:
                print(f"Failed to load CNN Model: {e}")
    
    def preprocess(self, img_pil):
        # Resize to 100x100, Normalize 0.5
        img = img_pil.resize((100, 100))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        img_np = np.transpose(img_np, (2, 0, 1))
        return torch.from_numpy(img_np).unsqueeze(0)

    def predict(self, face_pil):
        if not self.loaded: return "N/A", 0.0
        inp = self.preprocess(face_pil).to(self.device)
        with torch.no_grad():
            out = self.model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, 1)
            return ('Male' if idx.item() == 1 else 'Female'), conf.item()

# --- Manual ResNet Implementation (since torchvision is broken) ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(pretrained=False, **kwargs):
    # Ignore pretrained arg since we load our own weights
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

class ResNetStrategy(ModelStrategy):
    def __init__(self, path, device):
        self.device = device
        self.loaded = False
        
        # Now we don't rely on TORCHVISION_AVAIL
        if os.path.exists(path):
            try:
                # Use our manual ResNet18
                self.model = resnet18(num_classes=2) 
                self.model.load_state_dict(torch.load(path, map_location=device))
                self.model.float() # Ensure Float32
                self.model.to(device).eval()
                print(f"ResNet Dtype: {self.model.conv1.weight.dtype}") # DEBUG
                self.loaded = True
                print("ResNet18 loaded successfully.")
            except Exception as e: 
                print(f"ResNet Error: {e}")
        else:
            print(f"ResNet file not found at: {path}")
    
    def preprocess(self, img_pil):
        # ResNet ImageNet stats
        img = img_pil.resize((100, 100))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_np = (img_np - mean) / std
        img_np = np.transpose(img_np, (2, 0, 1))
        return torch.from_numpy(img_np).unsqueeze(0)

    def predict(self, face_pil):
        if not self.loaded: return "N/A", 0.0
        try:
            inp = self.preprocess(face_pil).to(self.device)
            # Force input type to match model type (fix for Double/Float mismatch)
            inp = inp.type_as(next(self.model.parameters()))
            
            with torch.no_grad():
                out = self.model(inp)
                probs = torch.nn.functional.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                return ('Male' if idx.item() == 1 else 'Female'), conf.item()
        except Exception as e:
            print(f"ResNet Crash: {e}")
            return "Error", 0.0

class EmbeddingStrategy(ModelStrategy):
    def __init__(self, path, device):
        self.device = device
        self.loaded = False
        if FACENET_AVAIL and os.path.exists(path):
            try:
                self.model = EmbeddingClassifier()
                self.model.load_state_dict(torch.load(path, map_location=device), strict=False)
                self.model.to(device).eval()
                self.loaded = True
                print("FaceNet loaded successfully.")
            except Exception as e:
                print(f"FaceNet Loading Error: {e}")
                pass

    def preprocess(self, img_pil):
        # 160x160, -1 to 1
        img = img_pil.resize((160, 160))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        img_np = np.transpose(img_np, (2, 0, 1))
        return torch.from_numpy(img_np).unsqueeze(0)

    def predict(self, face_pil):
        if not self.loaded: return "N/A", 0.0
        inp = self.preprocess(face_pil).to(self.device)
        with torch.no_grad():
            out = self.model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, 1)
            return ('Male' if idx.item() == 1 else 'Female'), conf.item()

class SVMStrategy(ModelStrategy):
    def __init__(self, path):
        self.model = None
        if SKIMAGE_AVAIL and os.path.exists(path):
            try:
                self.model = joblib.load(path)
            except: pass

    def predict(self, face_pil):
        if not self.model: return "N/A", 0.0
        # HOG
        img = np.array(face_pil.resize((64, 64)))
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2), channel_axis=-1)
        
        # Scikit-learn predict
        idx = self.model.predict([fd])[0]
        # Probability
        try:
            probs = self.model.predict_proba([fd])[0]
            conf = probs[idx]
        except: conf = 1.0 # If no probability
        
        return ('Male' if idx==1 else 'Female'), conf

class LogRegStrategy(ModelStrategy):
    def __init__(self, path):
        self.model = None
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
            except: pass
            
    def predict(self, face_pil):
        if not self.model: return "N/A", 0.0
        # Flatten
        img = np.array(face_pil.resize((64, 64)))
        flat = img.flatten()
        
        idx = self.model.predict([flat])[0]
        try:
            probs = self.model.predict_proba([flat])[0]
            conf = probs[idx]
        except: conf = 1.0
        
        return ('Male' if idx==1 else 'Female'), conf

# ==========================================
# 3. Predictor Engine
# ==========================================
class GenderPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Strategies
        self.strategies = {
            "Custom CNN": CNNStrategy('gender_cnn_model.pth', self.device),
            "ResNet18": ResNetStrategy('gender_resnet.pth', self.device),
            "Embedding (FaceNet)": EmbeddingStrategy('gender_embedding.pth', self.device),
            "SVM (HOG)": SVMStrategy('gender_svm.pkl'),
            "Logistic Regression": LogRegStrategy('gender_logreg.pkl')
        }
        self.current_model = "Custom CNN"

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_model(self, model_name):
        if model_name in self.strategies:
            self.current_model = model_name
            print(f"Switched to {model_name}")

    def predict(self, image):
        if isinstance(image, np.ndarray):
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
             image_pil = image

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        strategy = self.strategies[self.current_model]
        
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = x, y, x + w, y + h
            face_img = image_pil.crop((x1, y1, x2, y2))
            
            gender, conf = strategy.predict(face_img)
            
            results.append({
                'box': [x1, y1, x2, y2],
                'gender': gender,
                'confidence': conf
            })
        return results

# ==========================================
# 4. PySide6 GUI
# ==========================================
class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.Signal(np.ndarray)

    def __init__(self, predictor):
        super().__init__()
        self._run_flag = True
        self.predictor = predictor

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        last_results = []
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % 5 == 0:
                    last_results = self.predictor.predict(frame)
                
                for res in last_results:
                    x1, y1, x2, y2 = res['box']
                    gender = res['gender']
                    conf = res['confidence']
                    
                    color = (0, 255, 0) if gender == 'Male' else (255, 0, 255)
                    label = f"{gender} ({conf:.2f})"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                self.change_pixmap_signal.emit(frame)
            time.sleep(0.033) 
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gender Classifier - Multi-Model")
        self.resize(1000, 700)
        
        self.predictor = GenderPredictor()

        # Layouts
        self.main_layout = QtWidgets.QVBoxLayout()
        self.control_layout = QtWidgets.QHBoxLayout()

        # Model Selector
        self.model_label = QtWidgets.QLabel("Select Model:")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(self.predictor.strategies.keys())
        self.model_combo.currentTextChanged.connect(self.change_model)
        
        self.control_layout.addWidget(self.model_label)
        self.control_layout.addWidget(self.model_combo)

        # Buttons
        self.start_btn = QtWidgets.QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_video)
        self.control_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setDisabled(True)
        self.control_layout.addWidget(self.stop_btn)

        self.main_layout.addLayout(self.control_layout)

        # Video Label
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        self.setLayout(self.main_layout)
        self.thread = None

    def change_model(self, text):
        self.predictor.set_model(text)

    def start_video(self):
        self.thread = VideoThread(self.predictor)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.start_btn.setDisabled(True)
        self.stop_btn.setDisabled(False)

    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.start_btn.setDisabled(False)
        self.stop_btn.setDisabled(True)

    @QtCore.Slot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = App()
    window.show()
    sys.exit(app.exec())
