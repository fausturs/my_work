#include <iostream>
#include <fstream>
#include <utility>

#include "func.hpp"
#include "TF.hpp"
#include "sparse_tensor.hpp"

int main(int args, const char* argv[])
{
	/*
	auto tensor = create_tensor();
	wjy::save_sparse_tensor(tensor, "../data/tensor.txt");
	*/
    TF<3>::element_tp learning_rate=0.001;
    //if (args>1) learning_rate = std::atof(argv[1]);
	wjy::sparse_tensor<double, 3> t;
	wjy::load_sparse_tensor(t, "../data/tensor_dim3_20180405.txt");
	//double a = 0;
	//for (auto & x: t) a += x.second*x.second;
	//std::cout<<(long long)a<<std::endl; 
	TF<3> tf;
	tf.initialize({10, 10, 10}, 1000000, 1000, 1, 1, 0.5, learning_rate);
	tf.train(t, std::cout);
	tf.save("../model/20180408_2.mod");
    return 0;
}


























