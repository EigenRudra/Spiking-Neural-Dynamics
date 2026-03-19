#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip> // For clean terminal formatting

class SNNLayer 
{
private:
	double tau,vrest,vth,vreset,R,dt;        
	int n;	//No of neurons
	Eigen::VectorXd v,spikes;   
	Eigen::MatrixXd weights;  

public:
	SNNLayer(int n_inputs, int n, double dt) 
        {
        	//Standard values for biological systems
        	tau=10;     
		vrest=-65;   
		vth=-50;     
		vreset=-65;  
		R=1;
		this->n=n;
		this->dt=dt;
		v=Eigen::VectorXd::Constant(n,vrest);
		spikes=Eigen::VectorXd::Zero(n);
		weights=Eigen::MatrixXd::Random(n, n_inputs).cwiseAbs()*40.0;	//40 multiplier to ensure the random fraction carry enough current to overcome the -65mV leak
	}
	
	Eigen::VectorXd step(const Eigen::VectorXd& inSpikes)
	{
		Eigen::VectorXd I=weights*inSpikes;	//Input current
		Eigen::VectorXd dv= (dt/tau) * ( -(v.array()-vrest) + (R*I.array()) ).matrix();
		v+=dv;
		for(int i=0;i<n;i++)
		{
			if(v(i)>=vth)
			{
				spikes(i)=1;       
				v(i)=vreset;      
			}
			else
				spikes(i)=0;
		}
		return spikes;
	}

	Eigen::VectorXd voltages() const	//Not utilized in this main()
	{
		return v;
	}
	Eigen::MatrixXd getWeight() const 
	{ 
		return weights;
	}
};

int main()
{
	double dt=1,time=100;
	int inputs=5,neurons=3;
	SNNLayer layer(inputs,neurons,dt);
	std::cout<<"Initial Random Weight Matrix:\n"<<layer.getWeight()<<"\n";
	std::cout << "-----------------------------------\n";

	for(int t=0;t<time/dt;t++)
	{
		//Using prime numbers to create a complex, non-repeating interference pattern
		//Working on changing it to Poisson distribution
		Eigen::VectorXd inSpikes=Eigen::VectorXd::Zero(inputs);
		if(t%2==0) inSpikes(0)=1;	//500Hz
		if(t%3==0) inSpikes(1)=1;	//333 Hz
		if(t%5==0) inSpikes(2)=1.0;	//200 Hz
		if(t%7==0) inSpikes(3)=1.0;	//142 Hz
		if(t%11==0) inSpikes(4)=1.0;	//90 Hz

		Eigen::VectorXd outSpikes=layer.step(inSpikes);
		if(outSpikes.sum()>0)
		{
			//Printing spike vector to see which neuron fired
			std::cout<<"Time: "<<std::setw(3)<<t*dt<<" ms | SPIKES: ["<<outSpikes.transpose()<<"]\n";	//Taking transpose for better visual output in terminal
		}
	}
	return 0;
}
