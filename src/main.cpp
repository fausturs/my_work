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
    TF<3>::element_tp learning_rate=0.0001;
    if (args>1) learning_rate = std::atof(argv[1]);
	wjy::sparse_tensor<double, 3> t;
	wjy::load_sparse_tensor(t, "../data/tensor_dim3_20180405.txt");
	TF<3> tf;
	tf.initialize({10, 10, 10}, 200000, 1000, 1, 0.2, 0.5, learning_rate);
	tf.train(t, std::cout);
	tf.save("../a.txt");
    return 0;
}


























