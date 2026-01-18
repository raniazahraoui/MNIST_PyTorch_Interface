🧠 MNIST Handwriting Recognition (PyTorch + Tkinter)

Ce projet implémente une application de reconnaissance de chiffres manuscrits (0–9) basée sur le dataset MNIST, utilisant PyTorch pour le deep learning et Tkinter pour l’interface graphique.

L’utilisateur peut dessiner un chiffre à la souris, et le modèle prédit le chiffre avec un taux de confiance.

🚀 Fonctionnalités

Entraînement d’un réseau de neurones sur MNIST

Évaluation de la précision sur le jeu de test

Interface graphique interactive (dessin à la souris)

Prétraitement de l’image similaire à MNIST

Affichage du chiffre prédit et de la confiance

🏗️ Modèle utilisé

Réseau de neurones fully-connected :

Entrée : 28 × 28 pixels (784)

Couches cachées :

128 neurones + ReLU

64 neurones + ReLU

Sortie : 10 neurones (classes 0 à 9)

🛠️ Technologies

Python 3

PyTorch

Torchvision

NumPy

SciPy

Pillow (PIL)

Tkinter

📦 Installation

Installer les dépendances nécessaires :

pip install torch torchvision numpy scipy pillow


⚠️ Tkinter est inclus par défaut avec Python.

▶️ Exécution

Lancer le projet avec :

python main.py


Le programme :

Télécharge automatiquement le dataset MNIST

Entraîne le modèle pendant 15 epochs

Teste le modèle

Ouvre une interface graphique pour dessiner des chiffres

🎨 Interface graphique

Dessiner un chiffre avec la souris

Bouton Prédire : affiche le chiffre reconnu et la confiance

Bouton Effacer : nettoie le canvas

🧪 Prétraitement de l’image

L’image dessinée est :

Convertie en niveaux de gris

Recadrée automatiquement

Redimensionnée et centrée (28×28)

Normalisée comme les images MNIST

Convertie en tenseur PyTorch

📊 Exemple de sortie
Epoch [15/15], Loss: 0.0284
✅ Entraînement terminé
Accuracy sur test set: 97.8%
Chiffre prédit: 3
Confiance: 95.2%

🔧 Améliorations possibles

Remplacer le modèle par un CNN

Sauvegarder / charger le modèle entraîné

Ajouter des graphiques de performance

Améliorer le prétraitement du dessin

Déployer l’application en version web
