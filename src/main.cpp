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
#include "my_model_2.hpp"
#include "my_model_3.hpp"

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


std::string train_tensor_path       = "../data/20180515/tensor_dim4_80_20180515.txt";
std::string test_tensor_path        = "../data/20180515/tensor_dim4_20_20180515.txt";
std::string company_category_path   = "../data/20180515/category_of_company_80_20180515.txt";
int main(int args, const char* argv[])
{
    wjy::sparse_tensor<double ,4> train_tensor, test_tensor;
    wjy::load_sparse_tensor(train_tensor, train_tensor_path);
    wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    auto train_tensors = wjy::split_sparse_tensor(train_tensor, 3);
    std::vector< std::vector<double> > old_parameters;

    std::string file_path = "../model/20180515_1_";
    //  2013-2015 3 years
    int n = 3;
    for (int i=0; i<n; i++)
    {
        wjy::my_model_2<double, 3> mm2(std::move(train_tensors[i]), 10, 0.5, 1000, old_parameters, 0.1);
        mm2.train(sgd, std::cout, distribution1);
        mm2.train(gd, std::cout);
        old_parameters.push_back( std::move(mm2.get_parameters()) );
        mm2.save(file_path + std::to_string(i+2013)+".mod");
        break;
    }

    return 0;
}







