import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torchvision.transforms import v2


class NeuralNetwork(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input, 12, (3, 3)),                         #(batch_size, 1, 150, 150) => (batch_size, 12, 148, 148)
            nn.ReLU(),         
            nn.BatchNorm2d(12),                     
            nn.MaxPool2d(kernel_size=2, padding=0),               #(batch_size, 12, 148, 148) => (batch_size, 12, 74, 74)
            nn.Dropout2d(0.1),
            
            nn.Conv2d(12, 25, (7, 7)),                            #(batch_size, 12, 74, 74) => (batch_size, 25, 68, 68)
            nn.ReLU(),
            nn.BatchNorm2d(25),
            nn.MaxPool2d(kernel_size=2, padding=0),               #batch_size, 25, 68, 68) => (batch_size, 25, 34, 34)
            nn.Dropout2d(0.1),

            nn.Conv2d(25, 33, (7, 7)),                            #(batch_size, 25, 34, 34) => (batch_size, 33, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(33),
            nn.MaxPool2d(kernel_size=2, padding=0),               #(batch_size, 33, 28, 28) => (batch_size, 33, 14, 14)
            nn.Dropout2d(0.1),
            
            # nn.Flatten(),                                       #(batch_size, 33*14*14)
            nn.AdaptiveAvgPool2d((3, 3))
        )                                      

        self.classifier = nn.Sequential(
            nn.Linear(33*3*3, 150),
            # nn.BatchNorm1d(300),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(150, 75),
            # nn.BatchNorm1d(150),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(75, 30),
            # nn.BatchNorm1d(30),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(30, output),
            nn.Dropout(0.4),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.conv_layers(x)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat-Dog classifier")
        self.setWindowFlags(Qt.Window | Qt.MSWindowsFixedSizeDialogHint)
        self.setGeometry(100, 100, 800, 600)

        self.setAcceptDrops(True)  # Разрешаем перетаскивание файлов
        self.label = QLabel('Drop a photo of your pet here', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('border: 2px dashed #aaa; background-color: #f9f9f9;')
        self.label.setFixedWidth(500)
        
        self.cat, self.dog = 0, 0

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(os.listdir('models'))
        self.modelComboBox.setFixedWidth(250)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(150, 150), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        self.labelcat = QLabel(f'{0:.2f} cat', self)
        self.labelcat.setAlignment(Qt.AlignLeft)
        self.labelcat.setStyleSheet('font-size: 40px; color: gray;')

        self.labeldog = QLabel(f'{0:.2f} dog', self)
        self.labeldog.setAlignment(Qt.AlignLeft)
        self.labeldog.setStyleSheet('font-size: 40px; color: gray;') 

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.modelComboBox)
        rightLayout.addWidget(self.labelcat)
        rightLayout.addWidget(self.labeldog)
        rightLayout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addLayout(rightLayout)

        self.setLayout(mainLayout)

        self.modelComboBox.currentIndexChanged.connect(self.on_model_changed)


    def update_proba(self):
        self.labelcat.setText(f'{self.cat:.2f} cat')
        self.labeldog.setText(f' {self.dog:.2f} dog')


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()


    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    pixmap = QPixmap(file_path)
                    if not pixmap.isNull():
                        self.file_path_image = file_path
                        self.set_image(pixmap)
                        self.run_model(file_path)
                        self.update_proba()
                else:
                    self.label.setText("Invalid file format. Please drop an image.")


    def set_image(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        self.label.setStyleSheet('border: 2px dashed #aaa; background-color: #e1e1e1;')


    def run_model(self, path):
        self.file_path = self.modelComboBox.currentText()
        print(self.file_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.state = torch.load(f'models\{self.file_path}', weights_only=True)
            self.current_model = NeuralNetwork(3, 2).to(self.device)
            self.current_model.load_state_dict(self.state)
            self.current_model.eval()
            sample = np.array(Image.open(path))
            if sample.shape[2] == 4:
                sample = sample[:, :, :3]
            sample = self.transform(sample)
            sample = sample.unsqueeze(0).to(self.device)
            with torch.no_grad():
                current_pred = self.current_model(sample)
                probabilities = current_pred.softmax(dim=1).tolist()[0]
            self.cat, self.dog = probabilities
        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_model = None


    def on_model_changed(self):
        if hasattr(self, 'file_path'):
            self.run_model(self.file_path_image)
            self.update_proba()


if __name__ == "__main__":
    app=QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
