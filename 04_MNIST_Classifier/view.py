import numpy as np
import matplotlib.pyplot as plt
import os

weights_file = 'trained_weights.csv'
labels_file = 'neuron_labels.csv'

def visualize_digit():
	print("Loading the 400-neuron brain from disk...")
	if not os.path.exists(weights_file) or not os.path.exists(labels_file):
		print("Error: Could not find trained_weights.csv or neuron_labels.csv! ")
		print("Make sure you have run the SNN engine first!")
		return False

	weights=np.loadtxt(weights_file, delimiter=',')
	labels=np.loadtxt(labels_file, delimiter=',')
	
	while True:
		i=input("\nEnter a digit to visualize (0-9) or 'q' to quit: ").strip().lower()
		
		if i=='q':
			print("Program terminated!")
			break
		try:
			target=int(i)
			if not(0<=target<=9):
				print("Please enter a number between 0 and 9!")
				continue
		except ValueError:
			print("Invalid input! Enter a digit or 'q' ")
			continue
 
		#Find indices of all required neurons
		neuron_I=np.where(labels==target)[0]
		numNeurons=len(neuron_I)
		print(f"Found {numNeurons} neurons out of 400 specialized for the digit '{target}' ")
		if numNeurons==0:
			print(f"No neurons were assigned to '{target}' ")
			continue

		#Plotting
		cols=min(10, numNeurons)
		rows=int(np.ceil(numNeurons/cols))
		plt.figure(figsize=(cols*1.5,rows*1.5),facecolor='black')
		plt.suptitle(f"Neurons Specialized in Detecting '{target}'", fontsize=16, color='white')

		for i in range(numNeurons):
			idx=neuron_I[i]
			img=weights[idx].reshape(28,28)
			ax = plt.subplot(rows,cols,i+1)
			ax.imshow(img, cmap='hot',interpolation='nearest')
			ax.set_title(f"Neuron {idx}",fontsize=10,color='white')
			ax.axis('off')

		plt.tight_layout()
		print(f"Rendering window for Digit {target}...")
		plt.show()

if __name__ == "__main__":
	visualize_digit()
