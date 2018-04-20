#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <functional>

#include "trainer.hpp"
#include "GD.hpp"
#include "SGD.hpp"

#include "predictor.hpp"
#include "tensor_factorization_predictor.hpp"
#include "tucker_decomposition.hpp"
#include "canonical_decomposition.hpp"
#include "pairwise_interaction_tensor_factorization.hpp"

#include "func.hpp"

std::mt19937 mt1(1);
std::uniform_real_distribution<double> urd1(-1, 1);
auto distribution1 = std::bind(urd1, std::ref(mt1));

std::mt19937 mt2(1);
std::uniform_real_distribution<double> urd2(-2, 2);
auto distribution2 = std::bind(urd2, std::ref(mt2));

size_t gd_iter_num = 20;
size_t gd_epoch_size = 1;
double gd_learning_rate = 0.00002;
double gd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > gd = std::make_shared< wjy::GD<double> >(gd_iter_num, gd_epoch_size, gd_learning_rate, gd_convergence_condition);

int sgd_random_seed = 1;
size_t sgd_iter_num = 20000;
size_t sgd_epoch_size = 1000;
double sgd_learning_rate = 0.001;
double sgd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > sgd = std::make_shared< wjy::SGD<double> >(sgd_iter_num, sgd_epoch_size, sgd_learning_rate, sgd_convergence_condition, sgd_random_seed);

std::string train_tensor_path   = "../data/20180415/tensor_dim3_80_20180415.txt";
std::string test_tensor_path    = "../data/20180415/tensor_dim3_20_20180415.txt";

int main(int args, const char* argv[])
{
    wjy::sparse_tensor<double ,3> tensor;
    wjy::load_sparse_tensor(tensor, train_tensor_path);
//    wjy::tucker_decomposition<double, 3> td(std::move(tensor), {10, 10, 10}, 0.5, 100s0);
//    td.train(gd, std::cout, distribution1);
//    wjy::canonical_decomposition<double, 3> cd(std::move(tensor), 10, 0.5, 1000);
//    cd.train(gd, std::cout, distribution1);
    wjy::pairwise_interaction_tensor_factorization<double, 3> pitf(std::move(tensor), 10, 0.5, 1000);
    pitf.train(sgd, std::cout, distribution1);

    return 0;
}







