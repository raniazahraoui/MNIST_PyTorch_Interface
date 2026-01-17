import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy import ndimage

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = SimpleNet()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 15

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("✅ Entraînement terminé")


def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy sur test set: {100 * correct / total:.2f}%')

test_model()


def preprocess_image(img):
    """Prétraitement similaire à MNIST"""
    # Convertir en numpy array
    img_array = np.array(img)
    
    # Trouver la bounding box du chiffre
    rows = np.any(img_array > 30, axis=1)
    cols = np.any(img_array > 30, axis=0)
    
    if not rows.any() or not cols.any():
        # Image vide
        return np.zeros((28, 28))
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extraire le chiffre
    img_cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # Redimensionner en gardant le ratio
    h, w = img_cropped.shape
    size = max(h, w)
    
    # Créer une image carrée avec le chiffre centré
    squared = np.zeros((size, size))
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    squared[y_offset:y_offset+h, x_offset:x_offset+w] = img_cropped
    
    # Redimensionner à 20x20 (MNIST utilise 20x20 puis ajoute padding)
    from scipy.ndimage import zoom
    scale = 20.0 / size
    img_20 = zoom(squared, scale)
    
    # Centrer dans une image 28x28
    img_28 = np.zeros((28, 28))
    y_offset = (28 - 20) // 2
    x_offset = (28 - 20) // 2
    img_28[y_offset:y_offset+20, x_offset:x_offset+20] = img_20
    
    # Normaliser les valeurs
    img_28 = img_28 / 255.0
    
    return img_28



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Handwriting Recognition")
        self.geometry("300x400")

        self.label = tk.Label(self, text="Dessinez un chiffre (0-9)", font=("Arial", 12))
        self.label.pack()

        self.canvas = tk.Canvas(self, width=280, height=280, bg='black')
        self.canvas.pack()

        self.button_predict = tk.Button(self, text="Prédire", command=self.predict, bg='green', fg='white')
        self.button_predict.pack(pady=5)
        
        self.button_clear = tk.Button(self, text="Effacer", command=self.clear, bg='red', fg='white')
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (280, 280), color=0)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 12  # Rayon du pinceau
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', width=0)
        self.draw_image.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_image = ImageDraw.Draw(self.image)

    def predict(self):
        # Prétraiter l'image
        img_processed = preprocess_image(self.image)
        
        # Convertir en tensor PyTorch
        img_tensor = torch.from_numpy(img_processed).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        
        # Normaliser comme MNIST
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Prédiction
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)

        result_text = f"Chiffre prédit: {pred.item()}\nConfiance: {confidence.item()*100:.1f}%"
        print(result_text)
        messagebox.showinfo("Résultat", result_text)


app = App()
app.mainloop()