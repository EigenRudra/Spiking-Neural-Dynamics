import matplotlib.pyplot as plt

time=[]
voltages=[]
with open('output.txt','r') as file:
	for line in file:
		if '|' in line and 'Time' not in line:
			parts=line.split('|')
			if len(parts)>=2:
				try:
					t=float(parts[0].strip())
					v=float(parts[1].strip())
					time.append(t)
					voltages.append(v)
				except ValueError:
					pass

plt.figure(figsize=(10,5))
plt.plot(time, voltages, label='Membrane Potential ($V_m$)', color='blue', linewidth=2)

# Adding reference lines
plt.axhline(y=-50.0, color='red', linestyle='--', label='Threshold (-50 mV)')
plt.axhline(y=-65.0, color='green', linestyle='--', label='Rest/Reset (-65 mV)')


plt.title('Leaky Integrate-and-Fire (LIF) Neuron Dynamics',fontsize=14)
plt.xlabel('Time (ms)',fontsize=12)
plt.ylabel('Voltage (mV)',fontsize=12)
plt.legend(loc='upper left')
plt.grid(True,alpha=0.3)
plt.tight_layout()

plt.savefig('lif_graph.png',dpi=300)
plt.show()
