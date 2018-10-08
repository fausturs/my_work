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
#include "my_model_4.hpp"

#include "func.hpp"

std::ofstream myout;
std::ifstream myin;

std::mt19937 mt1(1);
std::uniform_real_distribution<double> urd1(-1, 1);
auto distribution1 = std::bind(urd1, std::ref(mt1));

std::mt19937 mt2(1);
std::uniform_real_distribution<double> urd2(-2, 2);
auto distribution2 = std::bind(urd2, std::ref(mt2));

size_t gd_iter_num = 20000;
size_t gd_epoch_size = 2000;
double gd_learning_rate = 0.000001;
double gd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > gd = std::make_shared< wjy::GD<double> >(gd_iter_num, gd_epoch_size, gd_learning_rate, gd_convergence_condition);

int sgd_random_seed = 1;
size_t sgd_iter_num = 200000;
size_t sgd_epoch_size = 20000;
double sgd_learning_rate = 0.001;
double sgd_convergence_condition = 0.001;
std::shared_ptr< wjy::trainer<double> > sgd = std::make_shared< wjy::SGD<double> >(sgd_iter_num, sgd_epoch_size, sgd_learning_rate, sgd_convergence_condition, sgd_random_seed);

std::string tensor_path             = "../data/20180906/tensor_dim4_20180906.txt";
std::string train_tensor_path       = "../data/20180906/tensor_dim4_80_20180906.txt";
std::string test_tensor_path        = "../data/20180906/tensor_dim4_20_20180906.txt";
std::string company_category_path   = "../data/20180906/category_of_company_80_20180906.txt";

int main(int args, const char* argv[])
{

    wjy::sparse_tensor<double ,4> train_tensor, test_tensor, tensor;
    // wjy::load_sparse_tensor(tensor, tensor_path);
    wjy::load_sparse_tensor(train_tensor, train_tensor_path);
    wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    auto train_tensors = wjy::split_sparse_tensor(train_tensor, 3);
    auto test_tensors = wjy::split_sparse_tensor(test_tensor, 3);
    std::vector< std::vector<double> > old_parameters;

    std::string file_path = "../model/20180913_1_";
    int n = 5;    //  2013-2017 5 years
    auto train_tensors = wjy::split_sparse_tensor(train_tensor, 3);
    
    std::vector<int> mini_batch_nums = {50, 500, 1000, 1000, 1000};

    for (int i=0; i<n; i++)
    {
        train_tensors[i][{24530, 11825, 680}] = 0;
        wjy::my_model_2<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i], old_parameters, 2);
        // wjy::pairwise_interaction_tensor_factorization<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i]);
        //wjy::my_model_1<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i]);
        // wjy::my_model_3<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i], old_parameters, 0.1);
        pred.train(sgd, std::cout, distribution1);
        pred.train(gd, std::cout);
        old_parameters.push_back( std::move(pred.get_parameters()) );
        pred.save(file_path + std::to_string(i+2013)+".mod");
    }
    
    wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_2<double, 3>();
    // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::pairwise_interaction_tensor_factorization<double, 3>();
    //wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_1<double, 3>();
    //年份分开计算各种指标
    // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_3<double, 3>();
    for (int i=0; i<n; i++)
    {
        pred->load(file_path + std::to_string(i+2013)+".mod");
        auto ae = all_evaluations(test_tensors[i], *pred);
        for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";
    }
    // print_top_k(tensor, 20);

    wjy::my_model_4< double, 3 > my_model(std::move(train_tensors[0]), 10, 0.5, 2000, 0.5, {}, 1, {}, 1, {});

    return 0;
}




/*
    20180821之前的main
 wjy::sparse_tensor<double ,4> train_tensor, test_tensor;
    wjy::load_sparse_tensor(train_tensor, train_tensor_path);
    wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    auto train_tensors = wjy::split_sparse_tensor(train_tensor, 3);
    auto test_tensors = wjy::split_sparse_tensor(test_tensor, 3);
    std::vector< std::vector<double> > old_parameters;
    std::vector< size_t > category;
    myin.open(company_category_path);
    int size = 0;
    myin>>size;category.resize(size);
    for (auto & x : category) myin>>x;
    myin.close();
    

    std::string file_path = "../model/20180714_1_";
    int n = 5;    //  2013-2017 5 years

    
    std::vector<int> mini_batch_nums = {50, 500, 1000, 1000, 1000};


 
    for (int i=0; i<n; i++)
    {
        train_tensors[i][{24599, 11825, 680}] = 0;
        //wjy::my_model_2<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i], old_parameters, 0.1);
        //wjy::pairwise_interaction_tensor_factorization<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i]);
        //wjy::my_model_1<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i]);
        wjy::my_model_3<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i], old_parameters, 0.1);
        pred.train(sgd, std::cout, distribution1);
        pred.train(gd, std::cout);
        old_parameters.push_back( std::move(pred.get_parameters()) );
        pred.save(file_path + std::to_string(i+2013)+".mod");
    }
    
    // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_2<double, 3>();
    //wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::pairwise_interaction_tensor_factorization<double, 3>();
    //wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_1<double, 3>();
    //年份分开计算各种指标
    wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_3<double, 3>();
    for (int i=0; i<n; i++)
    {
        pred->load(file_path + std::to_string(i+2013)+".mod");
        auto ae = all_evaluations(test_tensors[i], *pred);
        for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";
    }

    

    //所有年合在一块
    std::vector< wjy::predictor<double, wjy::sparse_tensor_index<3> > * > preds;
    // for (int i=0;i<n;i++) preds.push_back( new wjy::pairwise_interaction_tensor_factorization<double, 3>() );
    for (int i=0;i<n;i++) preds.push_back( new wjy::my_model_1<double, 3>() );
    // for (int i=0;i<n;i++) preds.push_back( new wjy::my_model_2<double, 3>() );
    // for (int i=0;i<n;i++) preds.push_back( new wjy::my_model_3<double, 3>() );

    for (int i=0; i<n; i++) preds[i]->load(file_path + std::to_string(i+2013)+".mod");

    auto ae = all_evaluations_1(test_tensors, preds);
    for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";


    for (int i=0;i<n;i++) std::cout<<train_tensors[i].size() + test_tensors[i].size()<<std::endl;
    */


