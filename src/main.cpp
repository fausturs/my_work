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
	wjy::sparse_tensor<double, 3> t;
	wjy::load_sparse_tensor(t, "../data/tensor_dim3_20180405.txt");
	TF<3> tf;
	tf.initialize({10, 10, 10});
	tf.train(t, std::cout);
	tf.save("../a.txt");
    return 0;
}


























