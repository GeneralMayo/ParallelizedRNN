#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <math.h>
#include "util.h"
#include "cycle_timer.h"
#include "omp.h"

using Eigen::MatrixXd;

#define XPATH "data/X_train.csv"
#define YPATH "data/y_train.csv"
#define IDX_TO_WORD_PATH "data/index_to_word.csv"

#define WORD_DIM 8000
#define HIDDEN_DIM 100
#define BTT_TRUNCATE 4


struct  Training_Data
{
	std::vector<std::vector<int>> X_train;
	std::vector<std::vector<int>> Y_train;
	std::vector<std::string> index_to_word;
};

struct Forward_Prop_Data
{
	MatrixXd O;
	MatrixXd S;
};

struct Gradients
{
	MatrixXd dLdU;
	MatrixXd dLdV;
	MatrixXd dLdW;
};

struct Weights
{
	MatrixXd U;
	MatrixXd V;
	MatrixXd W;
};

Weights init_weights(int hd, int wd){
	MatrixXd U = MatrixXd::Random(hd,wd).cast<double>()*(1.0/sqrt(wd));
	MatrixXd W = MatrixXd::Random(hd,hd)*(1.0/sqrt(hd));
	MatrixXd V = MatrixXd::Random(wd,hd)*(1.0/sqrt(hd));
	struct Weights weights;
	weights.U = U;
	weights.W = W;
	weights.V = V;

	return weights;
}

Gradients init_grads(){
	MatrixXd dLdU = MatrixXd::Zero(HIDDEN_DIM,WORD_DIM);
	MatrixXd dLdV = MatrixXd::Zero(WORD_DIM,HIDDEN_DIM);
	MatrixXd dLdW = MatrixXd::Zero(HIDDEN_DIM,HIDDEN_DIM);
	struct Gradients grads2;
	grads2.dLdU = dLdU;
	grads2.dLdV = dLdV;
	grads2.dLdW = dLdW;
	return grads2;
}


std::vector<std::vector<int>> get_X_Y_data(std::string path){
	std::vector<std::vector<int>> data;
	std::ifstream infile(path);

	while (infile) {
	    std::string s;
	    if (!getline(infile,s)) break;
	    
	    std::istringstream ss(s);
	    std::vector <int> record;

	    while (ss)
	    {
	      std::string s;
	      if (!getline( ss, s, ',' )) break;
	      record.push_back(stoi(s));
	    }

	    data.push_back(record);
	}
	return data;
}

std::vector<std::string> get_index_to_word(std::string path){
	std::vector<std::string> data;
	std::ifstream infile(path);

	std::string s;
	getline(infile,s);
	std::istringstream ss(s);

    while (ss)
    {
      std::string s;
      if (!getline( ss, s, ',' )) break;
      data.push_back(s);
    }
	return data;
}

Training_Data get_training_data(void){
	std::vector<std::vector<int>> X_train = get_X_Y_data(XPATH);
	std::vector<std::vector<int>> Y_train = get_X_Y_data(YPATH);
	std::vector<std::string> index_to_word = get_index_to_word(IDX_TO_WORD_PATH);
	struct Training_Data td;
	td.X_train = X_train;
	td.Y_train = Y_train;
	td.index_to_word = index_to_word;
	return td;
}

Forward_Prop_Data forward_propegation(std::vector<int> &x, Weights &weights){
	int T = x.size();
	MatrixXd s = MatrixXd::Zero(T,HIDDEN_DIM);
	MatrixXd o = MatrixXd::Zero(T,WORD_DIM);
	for(int t = 0; t<T; t++){
		if(t==0){
			for(int i = 0; i<HIDDEN_DIM; i++){
				s(t,i) = tanh(weights.U(i,x[t]));
			}
		} else {
			MatrixXd step(HIDDEN_DIM,1);
			step = step.cast<double>();
			step = weights.U.col(x[t]) + weights.W*(s.row(t-1).transpose());
			for(int i = 0; i<HIDDEN_DIM; i++){
				s(t,i) = tanh(step(i));
			}
		}
		MatrixXd step(WORD_DIM,1);
		step = step.cast<double>();
		step = weights.V*(s.row(t).transpose());
		step = exp(step.array());
		o.row(t) = (step/step.sum()).transpose();
	}

	struct Forward_Prop_Data fpd;
	fpd.S = s;
	fpd.O = o;
	return fpd;
}

void predict(std::vector<int> &x, Weights &weights){
	Forward_Prop_Data fpd = forward_propegation(x,weights);
	std::vector<int> predictions(x.size(),0);
	for(unsigned int i = 0; i<predictions.size(); i++){
		fpd.O.row(i).maxCoeff(&predictions[i]);
	}
	//return predictions;
}

double calculate_loss(std::vector<std::vector<int>> &x, std::vector<std::vector<int>> &y,
	Weights &weights, int numWords){
	double L = 0;
	//for each training sentence
	for(unsigned int i =0; i<y.size(); i++){
		Forward_Prop_Data fpd = forward_propegation(x[i],weights);
		MatrixXd O(x[i].size(),WORD_DIM);
		O = fpd.O;
		Eigen::VectorXf correct_predictions(x[i].size());
		//for each set of word predictions in sentence get P(correct word)
		for(unsigned int j = 0; j<x[i].size(); j++){
			correct_predictions(j) = O(j,y[i][j]);
		}
		L += -1 * log(correct_predictions.array()).sum();
	}
	return L/numWords;
}

void get_array_sum(Eigen::ArrayXf arr,std::string str){
	double sum = 0;
	for(int i = 0; i<arr.cols(); i++){
		sum += arr(i);
	}
	std::cout<<str<<sum<<std::endl;
}

void get_matrix_sum(Eigen::MatrixXd mat,std::string str){
	double sum = 0;
	for(int i = 0; i<mat.rows(); i++){
		for(int j = 0; j<mat.cols(); j++){	
			sum += mat(i,j);
		}
	}
	std::cout<<str<<sum<<std::endl;
}



void bptt(std::vector<int> &x, std::vector<int> &y, Weights &weights, int bptt_steps,
						Gradients &grads){
	//forward propegation
	Forward_Prop_Data fpd = forward_propegation(x, weights);

	MatrixXd delta_o(y.size(),WORD_DIM);
	delta_o = fpd.O;
	for(unsigned int i = 0; i<y.size(); i++){
		delta_o(i,y[i])-=1;
	}
	int prev_word;

	//reset gradients
	grads.dLdU = MatrixXd::Zero(HIDDEN_DIM,WORD_DIM);
	grads.dLdV = MatrixXd::Zero(WORD_DIM,HIDDEN_DIM);
	grads.dLdW = MatrixXd::Zero(HIDDEN_DIM,HIDDEN_DIM);

	//for each output backwards
	for(int i = y.size()-1; i>=0; i--){
		grads.dLdV += delta_o.row(i).transpose()*fpd.S.row(i);
		
		Eigen::Array<double,1,HIDDEN_DIM> step1;
		Eigen::Array<double,1,HIDDEN_DIM> step2;
		Eigen::Array<double,1,HIDDEN_DIM> delta_t;
		step1 = (weights.V.transpose()*delta_o.row(i).transpose()).transpose().array();
		step2 = 1-(fpd.S.row(i).array()*fpd.S.row(i).array());
		delta_t = step1.array() * step2;

		//back propegation through time (at most bptt_steps)
		for(int backProp = i; backProp>=std::max(0,i-bptt_steps); backProp--){
			if(backProp == 0){
				prev_word = y.size()-1;
			} else {
				prev_word = backProp -1;
			}

			grads.dLdW += delta_t.matrix().transpose()*fpd.S.row(prev_word);
			grads.dLdU.col(x[backProp])+= delta_t.matrix().cast<double>();
			step1 = (weights.W.transpose()*delta_t.matrix().transpose()).transpose();
			step2 = 1-(fpd.S.row(prev_word).array()*fpd.S.row(prev_word).array());
			delta_t = step1.array() * step2;
		}										
	}
}

void sgd_step(std::vector<int> &x, std::vector<int> &y, double learning_rate, Weights &cur_weights
				,int bptt_steps, Gradients &grads){
	//get gradients
	bptt( x, y, cur_weights, bptt_steps, grads);
	
	//updata weights
	cur_weights.U -= learning_rate * grads.dLdU;
    cur_weights.V -= learning_rate * grads.dLdV;
    cur_weights.W -= learning_rate * grads.dLdW;
}

double train_with_sgd(std::vector<std::vector<int>> &X_train, std::vector<std::vector<int>> &Y_train,
				 double learning_rate, Weights &cur_weights , int bptt_steps, int nepoch, 
				 int evaluate_loss_after, int num_words, Gradients &grads){
	double prev_loss = -1;
	double cur_loss = 0;
	
	Weights prev_weights = cur_weights;

	double total_time = 0;
	int t_start = CycleTimer::currentSeconds();
	printf("START_TRAINING\n");
	for(int epoch = 0; epoch<nepoch; epoch++){
		printf("epoch# %d\n",epoch);
		//check loss
		if(epoch % evaluate_loss_after == 0){
			total_time += CycleTimer::currentSeconds() - t_start;
			printf("Calculating Loss...\n");

			cur_loss = calculate_loss(X_train, Y_train, cur_weights, num_words);
			printf("Loss: %f\n",cur_loss);
			if(prev_loss != -1 && cur_loss > prev_loss){
				learning_rate = learning_rate*.5;
				printf("New Learning Rate: %f\n", learning_rate);
			}
			prev_loss = cur_loss;
			t_start = CycleTimer::currentSeconds();
		}

		//#pragma omp prallel
		for(unsigned int tr_ex = 0; tr_ex < X_train.size(); tr_ex++){
			//get new weights
			sgd_step(X_train[tr_ex], Y_train[tr_ex], learning_rate, cur_weights
				,bptt_steps, grads);
		}
	}
	total_time += CycleTimer::currentSeconds() - t_start;
	return total_time;
}


int main() {
	//Get Training Data
	struct Training_Data td = get_training_data();

	//Initialize Weights
	struct Weights weights = init_weights(HIDDEN_DIM,WORD_DIM);
	//Initialize Gradients
	struct Gradients grads = init_grads();

	int EXAMPLE_NUM = 100;
	int num_words = 0;
	for(int i = 0; i<EXAMPLE_NUM; i++){
		num_words+= td.Y_train[i].size();
	}

	std::vector<std::vector<int>>::const_iterator first_X = td.X_train.begin();
	std::vector<std::vector<int>>::const_iterator last_X = td.X_train.begin() + EXAMPLE_NUM;
	std::vector<std::vector<int>> X_train_sub(first_X, last_X);

	std::vector<std::vector<int>>::const_iterator first_Y = td.Y_train.begin();
	std::vector<std::vector<int>>::const_iterator last_Y = td.Y_train.begin() + EXAMPLE_NUM;
	std::vector<std::vector<int>> Y_train_sub(first_Y, last_Y);
	
	double LEARNING_RATE_INIT = .1;
	int BPTT_STEPS = 4;
	int NEPOCH = 20;
	int EVALUATE_LOSS_AFTER = 1;

	double total_time;
	total_time = train_with_sgd(X_train_sub, Y_train_sub, LEARNING_RATE_INIT,
				 weights , BPTT_STEPS,  NEPOCH, 
				 EVALUATE_LOSS_AFTER, num_words, grads);


	printf("total_time: %f\n", total_time);
	return(0);
}