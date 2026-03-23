import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading biological weight matrix...")
file_path="trained_weights.csv"

if not os.path.exists(file_path):
	print(f"Error: {file_path} not found!")
	exit()

#Loading 100x784 matrix
weights=np.loadtxt(file_path,delimiter=",")

#Displaying first 100
fig,axes=plt.subplots(10,10,figsize=(12, 12))
fig.canvas.manager.set_window_title('Spiking Neural Network - Synaptic Weights')
fig.suptitle("Unsupervised STDP Feature Detectors (100 Neurons)",fontsize=18,y=0.98)

for i, ax in enumerate(axes.flat):
	if i<len(weights):
		img=weights[i].reshape(28, 28)		#Reshaping 784 synaptic weights back into a 28x28 retina grid
		ax.imshow(img, cmap='hot', interpolation='nearest')

	ax.set_xticks([])
	ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(top=0.92)
print("Rendering brain state...")
plt.show()
