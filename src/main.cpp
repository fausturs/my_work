#include <iostream>
#include <fstream>
#include <utility>

#include "func.hpp"
#include "TF.hpp"
#include "sparse_tensor.hpp"

int main(int args, const char* argv[])
{
    //test();
    //create_tensor();
    wjy::sparse_tensor<double, 3> tensor;
    wjy::load_sparse_tensor(tensor, "../data/20180415/tensor_dim3_80_20180415.txt");
	/*
	auto p = split_tensor(tensor, 0.2);
	wjy::save_sparse_tensor(p.first, "../data/20180415/tensor_dim3_20_20180415.txt");
	wjy::save_sparse_tensor(p.second, "../data/20180415/tensor_dim3_80_20180415.txt");
	*/
	
	double learning_rate = 0.001;
	if (args>1) learning_rate = std::atof( argv[1] );
	TF<3> tf;
	tf.load("../model/20180416_3.mod");
	tf.initialize({10, 10, 10}, 200, 5, 1000, 1, 1, 0.5, learning_rate);
	tf.train(tensor, false, std::cout);	
	tf.save("../model/20180416_4.mod");	
	
    return 0;
}


























