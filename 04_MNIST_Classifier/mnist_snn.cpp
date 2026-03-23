#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <random>
#include <cmath>
#include <algorithm>

//MNIST binary file parser
uint32_t swap_endian(uint32_t val)
{
	return ((val<<24)&0xff000000) | ((val<<8)&0x00ff0000) | ((val>>8)&0x0000ff00) | ((val>>24)&0x000000ff);
}

std::vector<std::vector<uint8_t>> readImages(const std::string& path,int numImages)
{
	std::ifstream file(path, std::ios::binary);
	if(!file.is_open())
	{
		std::cerr<<"Cannot open "<<path<<"\n";
		exit(1);
	}
	uint32_t magic,num,rows,cols;
	file.read((char*)&magic,4); 
	file.read((char*)&num,4);
	file.read((char*)&rows,4);
	file.read((char*)&cols,4);
    
	std::vector<std::vector<uint8_t>> images(numImages,std::vector<uint8_t>(784));
	for(int i=0;i<numImages;i++) 
		file.read((char*)images[i].data(),784);
	return images;
}

std::vector<uint8_t> readLabels(const std::string& path,int numLabels)
{
	std::ifstream file(path, std::ios::binary);
	uint32_t magic,num;
	file.read((char*)&magic, 4);
	file.read((char*)&num, 4);
    
	std::vector<uint8_t> labels(numLabels);
	file.read((char*)labels.data(), numLabels);
	return labels;
}


//32-bit SNN engine
class STDPLayer
{
public:
	int n,inputs;
	float tau,vrest,vreset,R,dt;
	Eigen::VectorXf v;
	Eigen::VectorXf vth; 
	Eigen::VectorXf spikes;
	Eigen::MatrixXf weights;
    
	Eigen::VectorXf lastInspikes;
	Eigen::VectorXf lastOutspikes;
    
	//Tuned parameters
	float Aplus=0.05f,Aminus=0.025f;       
	float Tplus=5.0f,Tminus=20.0f;		//Tau plus and tau minus 
	float w_max=1.0f, w_min=0.0f;
	float thetaPlus=20.0f,thetaDecay=0.001f; 

	STDPLayer(int inputs,int n)
	{
        	this->inputs=inputs;
        	this->n=n;
        	
        	//Standard Biological values
		tau=20.0f;
		vrest=-65.0f;
		vreset=-65.0f;
		R=5.0f;
		dt=1.0f;
		v=Eigen::VectorXf::Constant(n,vrest);
		vth=Eigen::VectorXf::Constant(n,-50.0f); 
		spikes=Eigen::VectorXf::Zero(n);
		weights=Eigen::MatrixXf::Random(n,inputs).cwiseAbs()*0.2f;
        
		lastInspikes=Eigen::VectorXf::Constant(inputs,-1000.0f);
		lastOutspikes=Eigen::VectorXf::Constant(n,-1000.0f);
	}

	Eigen::VectorXf step(const Eigen::VectorXf& inSpikes,float currTime,bool isTraining)
	{
		for(int i=0;i<inputs;i++)
		{
			if(inSpikes(i)>0.5f)
				lastInspikes(i)=currTime;
		}

		Eigen::VectorXf I=weights*inSpikes;
		Eigen::VectorXf dv=(dt/tau)*( -(v.array()-vrest) + (R*I.array()) ).matrix();
		v+=dv;

		int winner=-1;
		float maxV=-1000.0f;
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
			spikes(winner)=1.0f;
			v(winner)=vreset; 
			lastOutspikes(winner)=currTime;
			for(int i=0;i<n;i++)
			{
				if(i!=winner) 
					v(i)=vreset;
			}
		}

		if(isTraining)
		{
			if(winner!=-1)
				vth(winner)+=thetaPlus;
			vth=(vth.array()-thetaDecay).cwiseMax(-62.0f).matrix();

			if(winner!=-1) 
			{
				for(int i=0;i<inputs;i++)
				{
					float dtp=ctime-lastInspikes(i);	//dtp=dt post
					if(dtp>=0 && dtp<50.0f)
						weights(winner,i) += Aplus*std::exp(-dtp/Tplus);
					else if(dtp>-50.0f && dtp<0)
						weights(winner,i) -= Aminus*std::exp(dtp/Tminus);
				}
				weights=weights.cwiseMax(w_min).cwiseMin(w_max);	//Clipping
			}
			for(int i=0;i<n;i++)
			{
				float sum=weights.row(i).sum();
				if(sum>0)
					weights.row(i)*=(78.0f/sum);
			}
        	}
        return spikes;
	}
};

//400 neurons, 1 epoch, 60k images training (75s flashing), 10k images testing (150s flashing), labelling (150s flashing)
int main()
{
	Eigen::initParallel();
	int num_train=60000,num_test=10000,num_neurons=400,num_epochs=1;
    
	std::cout<<"Loading 70,000 MNIST Images into RAM...\n";
	auto train_images=readImages("train-images-idx3-ubyte",num_train);
	auto train_labels=readLabels("train-labels-idx1-ubyte",num_train);
	auto test_images=readImages("t10k-images-idx3-ubyte",num_test);
	auto test_labels=readLabels("t10k-labels-idx1-ubyte",num_test);

	STDPLayer network(784,num_neurons);
	std::default_random_engine generator(42); 
	std::uniform_real_distribution<float> distribution(0.0f,1.0f);

	std::cout<<"\n[Phase 1] STDP Training:\n";
	float globalTime=0.0f;
	for(int epoch=0;epoch<num_epochs;epoch++)
	{
		std::cout<<"\n               EPOCH "<<epoch+1<<" / "<<num_epochs<<"\n";
		std::cout<<"=========================================\n";
        
        	for(int i=0;i<num_train;i++)
        	{
			if(i%2000==0)
				std::cout<<"Epoch "<<epoch+1<<" Progress: "<<i<<" / "<<num_train<<"\n";
            
			//Flashing for 75ms
			for(int t=0;t<75;t++)
			{
				Eigen::VectorXf inSpikes=Eigen::VectorXf::Zero(784);
				for(int j=0;j<784;j++)
				{
					if(distribution(generator) < (train_images[i][j]/255.0f) * 0.2f)
						inSpikes(j) = 1.0f;
				}
				network.step(inSpikes,globalTime,true);
				globalTime+=1.0f;
			}
			for(int t=0;t<15;t++)
			{
				network.step(Eigen::VectorXf::Zero(784), global_time, false);
				global_time+=1.0f;
			}
		}
	}
	network.vth=Eigen::VectorXf::Constant(num_neurons,-50.0f);



	std::cout<<"\n[Phase 2] Assigning classes to 400 specialized feature detectors:\n";
	std::vector<std::vector<int>> tallies(num_neurons,std::vector<int>(10, 0));
	
	for(int i=0;i<num_train;i++)	//i is image index here
	{
		if(i%10000==0)
			std::cout<<"Labeling Progress: " <<i<< " / " << num_train << "\n";
		std::vector<int> spikeCount(num_neurons, 0);
        
		//Flashing for 150ms
		for(int t=0;t<150;t++)
		{
			Eigen::VectorXf inSpikes=Eigen::VectorXf::Zero(784);
			for(int j=0;j<784;j++) 
			{
				if(distribution(generator) < (train_images[i][j]/255.0f) * 0.2f) 
					inSpikes(j)=1.0f;
			}
			Eigen::VectorXf out = network.step(inSpikes,globalTime,false);
			for(int k=0;k<num_neurons;k++)
				if(out(k)>0) 
					spikeCount[k]++;
			globalTime+=1.0f;
		}
		int label=train_labels[i];
		for(int j=0;j<num_neurons;j++)
			tallies[j][label] += spikeCount[j];
	}

	std::vector<int> neurons(num_neurons, -1);
	for(int i=0;i<num_neurons;i++) 
	{
		int best=-1,maxSpikes=0;
		for(int d=0;d<10;d++)
		{
			if(tallies[i][d]>max_spikes)
			{
				maxSpikes=tallies[i][d];
				best=d;
			}
		}
		neurons[i]=best;
	}
	
	std::cout<<"\n[Phase 3] Testing Accuracy on 10,000 unseen images:\n";
	int correct=0;
	for(int img=0;img<num_test;img++)
	{
		if(img%2000==0)
			std::cout<<"Testing Progress: "<<img<<" / "<<num_test<<"\n";
		std::vector<int> spikeCount(num_neurons,0);
        
		//Flashing for 150ms
		for(int t=0;t<150;t++)
		{
			Eigen::VectorXf inSpikes=Eigen::VectorXf::Zero(784);
			for(int j=0;j<784;j++)
			{
				if(distribution(generator) < (test_images[img][j]/255.0f) * 0.2f)
					inSpikes(j)=1.0f;
			}
			Eigen::VectorXf out=network.step(inSpikes,globalTime,false);
			for(int i=0;i<num_neurons;i++)
				if(out(i)>0)
					spikeCount[i]++;
			globalTime += 1.0f;
		}
        
		int maxSpikes=-1,predicted=-1;
		for(int i=0;i<num_neurons;i++)
		{
			if(spikeCounts[i]>maxSpikes && neuron[i]!=-1)
			{
				maxSpikes=spikeCount[i];
				predicted=neurons[i];
			}
		}
		if(predicted==test_labels[img])
			correct++;
	}

	std::cout<<"\n=============================================\n";
	std::cout << "FINAL SCALED ACCURACY: "<<(float)correct/num_test * 100.0f << "%\n";
	std::cout<<"==============================================\n";

	//Saving model for visualization and also to avoid running Phase 1 every time
	std::cout<<"\nSaving the trained brain...\n";
	std::ofstream weight_file("trained_weights.csv");
	for(int i=0;i<num_neurons;i++)
	{
		for(int j=0;j<784;j++)
		{
			weight_file<<network.weights(i,j);
			if(j<783)
				weight_file << ",";
		}
		weight_file << "\n";
	}
	weight_file.close();

	std::ofstream label_file("neuron_labels.csv");
	for(int i=0;i<num_neurons;i++)
	{
		label_file<<neuron_assignments[i];
		if(i<num_neurons-1)
			label_file << ",";
	}
	label_file.close();
	std::cout<<"Model successfully saved!\n";
	return 0;
}
