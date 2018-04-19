#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <functional>

#include "trainer.hpp"
#include "GD.hpp"
#include "SGD.hpp"
#include "predictor.hpp"
#include "linear_regression.hpp"
#include "tensor_factorization_predictor.hpp"
#include "tucker_decomposition.hpp"


std::mt19937 mt1(1);
std::uniform_real_distribution<double> urd1(-1, 1);
auto distribution1 = std::bind(urd1, std::ref(mt1));

std::mt19937 mt2(1);
std::uniform_real_distribution<double> urd2(-2, 2);
auto distribution2 = std::bind(urd2, std::ref(mt2));

size_t gd_iter_num = 2000;
size_t gd_epoch_size = 10;
double gd_learning_rate = 0.01;
double gd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > gd = std::make_shared< wjy::GD<double> >(gd_iter_num, gd_epoch_size, gd_learning_rate, gd_convergence_condition);

int sgd_random_seed = 1;
size_t sgd_iter_num = 20000;
size_t sgd_epoch_size = 100;
double sgd_learning_rate = 0.001;
double sgd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > sgd = std::make_shared< wjy::SGD<double> >(sgd_iter_num, sgd_epoch_size, sgd_learning_rate, sgd_convergence_condition, sgd_random_seed);

int main(int args, const char* argv[])
{
    wjy::sparse_tensor<double ,3> tensor;
    wjy::load_sparse_tensor(tensor, "../data/test_tensor_dim3_2.txt");
    wjy::tucker_decomposition<double, 3> td(std::move(tensor), {10, 10, 10}, 0.5, 20);
    

    td.train(sgd, std::cout, distribution1);

    return 0;
}


























