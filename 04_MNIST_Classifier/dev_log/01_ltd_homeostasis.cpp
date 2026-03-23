#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <cmath>

class STDPLayer 
{
private:
	double tau,vrest,vreset,R,dt;
	double vth_base,theta_plus,tau_th;     

	int n,inputs;
	Eigen::VectorXd v,spikes;
	Eigen::MatrixXd weights;
	Eigen::VectorXd lastInspikes,lastOutspikes;
	Eigen::VectorXd vth;                     

	double Aplus,Aminus,Tplus,Tminus,wmax,wmin;	///wmax and wmin are boundary weights for preventing <0 and infinity for synapses

public:
	STDPLayer(int inputs,int n,double dt) 
	{
		tau=10;
		vrest=-65;
		vreset=-65;
		R=5;	//High resistance Rfor coincidence detection
		this->n=n;
		this->inputs=inputs;
		this->dt=dt;
		v=Eigen::VectorXd::Constant(n,vrest);
		spikes=Eigen::VectorXd::Zero(n);
		
		vth_base=-50;		//The resting baseline threshold
		theta_plus=2;
		tau_th=500;
		vth=Eigen::VectorXd::Constant(n,vth_base);
		
		weights=Eigen::MatrixXd::Random(n, inputs).cwiseAbs()*10 + Eigen::MatrixXd::Constant(n,inputs,10);
		lastInspikes = Eigen::VectorXd::Constant(inputs, -1000.0);
		lastOutspikes = Eigen::VectorXd::Constant(n, -1000.0);
		
		Aplus=2.5;       
		Aminus=2;	//Slightly numerically smaller than A+ to keep bounded
		Tplus=2;
		Tminus=20;
		wmax=20;       
		wmin=0;     
	}
    
	Eigen::VectorXd step(const Eigen::VectorXd& inSpikes,double ctime)
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
					if(delta>-100) 
						weights(j, i) -= Aminus*std::exp(delta/Tminus);
				}
			}
		}

		//Solving LIF Membrane DE
		Eigen::VectorXd I=weights*inSpikes; 
		Eigen::VectorXd dv=(dt/tau) * (-(v.array()-vrest) + (R*I.array()) ).matrix();
		v+=dv;
        
		//Solving Homeostasis Threshold DE (Exponential Decay)
		Eigen::VectorXd dvth=(dt/tau_th)*(vth_base-vth.array()).matrix();
		vth+=dvth;

		//Lateral Inhibition & Spiking
		double maxV=-1000;
		int winner=-1;  
        
		for(int i=0;i<n;i++)
		{
			if(v(i)>=vth(i) && v(i)>maxV) 
			{
				maxV=v(i);
				winner=i;
			}
		}
		spikes.setZero();
		if(winner!=-1)
		{
			spikes(winner)=1;        
			v(winner)=vreset;      
			lastOutspikes(winner)=ctime;
			
			//Homeostasis
			vth(winner)+=theta_plus;
			for(int i=0;i<inputs;i++)
			{
				delta=ctime-lastInspikes(i); 
				if(delta>=0 && delta<100)
					weights(winner, i) += Aplus*std::exp(-delta/Tplus);
			}
			
			//Lateral Inhibition
			for(int i=0;i<n;i++) 
			{
				if(i!=winner)
					v(i)=vreset;
			}
		}

		//Clipping
		weights=weights.cwiseMax(wmin).cwiseMin(wmax);
		return spikes;
	}

	Eigen::MatrixXd getWeights() const
	{	
		return weights;
	}
	
	Eigen::VectorXd getThresholds() const
	{
		return vth;
	}
};

int main()
{
	double dt=1,time=3000,ct;
	int inputs=5,neurons=3;
    
	STDPLayer layer(inputs,neurons,dt);
	std::cout<<"Initial Random Weight Matrix:\n"<<layer.getWeights()<<"\n";
	
	std::cout<<"\nTraining network for "<<time<<" ms:\n";

	for(int i=0;i<time/dt;i++)
	{
		ct=i*dt;
		Eigen::VectorXd inSpikes=Eigen::VectorXd::Zero(inputs);
		if(i%20==0) 
			inSpikes(0)=inSpikes(1)=1;
		if(i%17==0)
			inSpikes(2)=1;   
		if(i%23==0)
			inSpikes(3)=1;   
		if(i%29==0)
			inSpikes(4)=1;   	
		Eigen::VectorXd outSpikes=layer.step(inSpikes,ct);
	}
	
	std::cout << "\nFinal Adapted Weight Matrix:\n" << layer.getWeights() << "\n";
	std::cout << "\nFinal Adaptive Thresholds:\n" << layer.getThresholds().transpose() << "\n";
    
	return 0;
}
