#include <iostream>
#include <fstream>
#include <utility>

#include "func.hpp"
#include "TF.hpp"
#include "sparse_tensor.hpp"

int main(int args, const char* argv[])
{
    test();
    create_tensor();
    wjy::sparse_tensor<double, 3> tensor;
    wjy::load_sparse_tensor(tensor, "../data/20180415/tensor_dim3_20180415.txt");
    return 0;
}


























