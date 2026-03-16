#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

class LIFNeuron
{
private:
	double tau;		//Membrane time constant
	double vrest;		//Resting potention
	double vth;		//Threshold potential to trigger spike
	double vreset;		//Potential the neuron resets to after spiking
	double R;		//Membrane resistance
	double v;		//Current membrane potential V(t)

public:
	LIFNeuron(double tau,double vrest,double vth,double vreset,double R)
	{
		this->tau=tau;
		this->vrest=vrest;
		this->vth=vth;
		this->vreset=vreset;
		this->R=R;
		v=vrest;
	}
	
	bool update(double I,double dt)
	{
		double drv=(-(v-vrest)+R*I)/tau;	//drv= derivative = dV/dt
		v+=drv*dt;
		if(v>=vth)
		{
			v=vreset;
			return true;
		}
		return false;
	}
	
	double voltage() const
	{
		return v;
	}
};

int main()
{
	double time=100,dt=1.0,tau,vrest,vth,vreset,R,I;
	std::cout<<"Enter Membrane Time Constant (in ms): ";
	std::cin>>tau;
	std::cout<<"Enter Resting Potential (in mV): ";
	std::cin>>vrest;
	std::cout<<"Enter Spike Threshold (in mV): ";
	std::cin>>vth;
	std::cout<<"Enter Reset Potential (in mV): ";
	std::cin>>vreset;
	std::cout<<"Enter Membrane Resistance (in MΩ): ";
	std::cin>>R;
	std::cout<<"Enter Constant Injected Current (in nA): ";
	std::cin>>I;

	LIFNeuron n(tau,vrest,vth,vreset,R);
	std::ofstream file("output.txt");
	std::cout<<"Time(ms) | Voltage(mV) | Event\n";
	std::cout<<"--------------------------------\n";
	file<<"Time(ms) | Voltage(mV) | Event\n";
    	file<<"--------------------------------\n";
	
	for(int i=0;i<time/dt;i++)
	{
		bool spike=n.update(I,dt);
		std::cout << std::setw(8) << (i*dt) << " | " << std::setw(11) << n.voltage() << " | ";    
		if(spike)
			std::cout << "SPIKE!";
		std::cout<<"\n";
		
		file << std::setw(8) << (i*dt) << " | " << std::setw(11) << n.voltage() << " | "; 
		if(spike) 
			file << "SPIKE!";
		file << "\n";
	}
	return 0;
}
