#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <cmath>

class STDPLayer 
{
private:
	double tau,vrest,vth,vreset,R,dt;        
	int n,inputs;
	Eigen::VectorXd v,spikes;
	Eigen::MatrixXd weights;
	Eigen::VectorXd lastInspikes,lastOutspikes;
	
	double Aplus,Aminus,Tplus,Tminus,wmax,wmin;	//wmax and wmin are boundary weights for preventing <0 and infinity for synapses
							//Tplus and Tminus are tau + and tau - respectively

public:
	STDPLayer(int inputs, int n, double dt) 
	{
		tau=10,vrest=-65,vth=-50,vreset=-65,R=5;         // High resistance Rfor coincidence detection
		this->n=n;
		this->inputs=inputs;
		this->dt=dt;
		v=Eigen::VectorXd::Constant(n,vrest);
		spikes=Eigen::VectorXd::Zero(n);
		
		weights=Eigen::MatrixXd::Random(n,inputs).cwiseAbs()*10 + Eigen::MatrixXd::Constant(n,inputs,10);
		lastInspikes=Eigen::VectorXd::Constant(inputs,-1000);
		lastOutspikes=Eigen::VectorXd::Constant(n,-1000);
		
		Aplus=2.5;       
		Aminus=2;	//Slightly numerically smaller than A+ to keep bounded
		Tplus=2;	//Reward window
		Tminus=20;	//Punishment window higher to catch all noise
		wmax=20;       
		wmin=0;     
	}
    
	Eigen::VectorXd step(const Eigen::VectorXd& inSpikes, double ctime)	//ctime=current time
	{
		double delta;
		//LTD 
		for(int i=0;i<inputs;i++)
		{
			if(inSpikes(i)>0.5)
			{
				lastInspikes(i)=ctime;
				for(int j=0;j<n;j++)
				{
					delta=lastOutspikes(j)-ctime; 
					if(delta>-100.0) 
						weights(j,i) -= Aminus*std::exp(delta/Tminus);
				}
			}
		}

		//Solving DE
		Eigen::VectorXd I=weights*inSpikes; 
		Eigen::VectorXd dv= (dt/tau)*( -(v.array()-vrest)+(R*I.array()) ).matrix();
		v+=dv;
        
		//LTP
		for(int i=0;i<n;i++)
		{
			if(v(i)>=vth)
			{
				spikes(i)=1;       
				v(i)=vreset;      
				lastOutspikes(i)=ctime;

				for(int j=0;j<inputs;j++) 
				{
					delta=ctime - lastInspikes(j); 
					if(delta>=0 && delta<100)
						weights(i,j) += Aplus*std::exp(-delta/Tplus);
				}
			}
			else
				spikes(i)=0;
		}

		//Clipping with max and min weights
		weights=weights.cwiseMax(wmin).cwiseMin(wmax);
		return spikes;
	}

	Eigen::MatrixXd getWeights() const 
	{
		return weights;
	}
};

int main()
{
	double dt=1,time=3000,ct;
	int inputs=5,neurons=3;
    
	STDPLayer layer(inputs,neurons,dt);
	std::cout<<"Initial Random Weight Matrix:\n"<<layer.getWeights()<<"\n";
	std::cout<<"Training network for "<<time<<" ms:\n\n";

	for(int i=0;i<time/dt;i++)
	{
		ct=i*dt;
		Eigen::VectorXd inSpikes=Eigen::VectorXd::Zero(inputs);
		if(i%20==0)
			inSpikes(0)=inSpikes(1)=1;
		//Using prime numbers for mimicking out of sync input such that they do not allow neuron to cross threshold voltage
		if(i%17==0)
			inSpikes(2)=1;   
		if(i%23==0)
			inSpikes(3)=1;   
		if(i%29==0)
			inSpikes(4)=1;   
		Eigen::VectorXd outSpikes=layer.step(inSpikes,ct);
	}
	std::cout<<"\nFinal Adapted Weight Matrix (Post-STDP):\n"<<layer.getWeights()<<"\n";
    
	return 0;
}
