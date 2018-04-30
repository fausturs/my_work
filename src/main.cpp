#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <functional>

#include "trainer.hpp"
#include "GD.hpp"
#include "SGD.hpp"
#include "AGD.hpp"

#include "predictor.hpp"
#include "tensor_factorization_predictor.hpp"
#include "tucker_decomposition.hpp"
#include "canonical_decomposition.hpp"
#include "pairwise_interaction_tensor_factorization.hpp"

#include "my_model_1.hpp"

#include "func.hpp"

std::ofstream myout;

std::mt19937 mt1(1);
std::uniform_real_distribution<double> urd1(-1, 1);
auto distribution1 = std::bind(urd1, std::ref(mt1));

std::mt19937 mt2(1);
std::uniform_real_distribution<double> urd2(-2, 2);
auto distribution2 = std::bind(urd2, std::ref(mt2));

size_t gd_iter_num = 20000;
size_t gd_epoch_size = 2000;
double gd_learning_rate = 0.000002;
double gd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > gd = std::make_shared< wjy::GD<double> >(gd_iter_num, gd_epoch_size, gd_learning_rate, gd_convergence_condition);

int sgd_random_seed = 1;
size_t sgd_iter_num = 200000;
size_t sgd_epoch_size = 20000;
double sgd_learning_rate = 0.001;
double sgd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > sgd = std::make_shared< wjy::SGD<double> >(sgd_iter_num, sgd_epoch_size, sgd_learning_rate, sgd_convergence_condition, sgd_random_seed);

// size_t agd_iter_num = 1000;
// size_t agd_epoch_size = 10;
// double agd_learning_rate = 0.000002;
// double agd_convergence_condition = 0.001;
// double agd_theta = 0.8;
// std::shared_ptr< wjy::trainer<double> > agd = std::make_shared< wjy::AGD<double> >(agd_iter_num, agd_epoch_size, agd_learning_rate, agd_convergence_condition, agd_theta);


std::string train_tensor_path       = "../data/20180415/tensor_dim3_80_20180415.txt";
std::string test_tensor_path        = "../data/20180415/tensor_dim3_20_20180415.txt";
std::string negative_tensor_path    = "../data/20180415/tensor_dim3_negative_20180415.txt";
std::string company_category_path   = "../data/20180415/category_of_company_80_20180415.txt";
int main(int args, const char* argv[])
{
    wjy::sparse_tensor<double ,3> train_tensor, test_tensor, negative_tensor;
    wjy::load_sparse_tensor(train_tensor, train_tensor_path);
    // wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    // wjy::load_sparse_tensor(negative_tensor, negative_tensor_path);

    // wjy::sparse_tensor<double ,3> tensor = std::move(train_tensor);
    // tensor.insert(negative_tensor.begin(), negative_tensor.end());

    // wjy::tucker_decomposition<double, 3> td(std::move(tensor), {10, 10, 10}, 0.5, 100s0);
    // wjy::canonical_decomposition<double, 3> cd(std::move(tensor), 10, 0.5, 1000);
    // wjy::pairwise_interaction_tensor_factorization<double, 3> pitf(std::move(tensor), 10, 0.5, 1000);

    // td.train(sgd, std::cout, distribution1);
    // td.train(gd, std::cout);
    // cd.train(sgd, std::cout, distribution1);
    // cd.train(gd, std::cout);
    // pitf.train(sgd, std::cout, distribution1);
    // pitf.train(gd, std::cout);

    std::ifstream myin(company_category_path);
    size_t n = 0; 
    myin>>n;
    std::vector<size_t> category_map(n);
    for(auto & c : category_map) myin>>c;
    myin.close();

    wjy::my_model_1<double, 3> mm1(std::move(train_tensor), 10, 0.5, category_map, 0.1, 1000);

    mm1.train(sgd, std::cout, distribution1);
    mm1.train(gd, std::cout);

    myout.open("../model/20180430_1.mod");
    mm1.save_parameters(myout);
    myout.close();
    
    return 0;
}







