#include <iostream>
#include <fstream>
#include <utility>
#include <random>
#include <functional>
#include <map>

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
#include "print.hpp"

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
std::string company_category_path   = "../data/20180906/category_of_company.txt";
std::string skill_category_path     = "../data/20180906/skill_category.txt";


std::map< size_t, std::vector< std::pair<size_t, double> > > skill_category_map;
void read_skill_category()
{
    std::ifstream myin(skill_category_path);
    assert(myin);
    skill_category_map.clear();
    int n, skill, m;
    myin>>n;
    for (int i=0; i<n; i++)
    {
        myin>>skill>>m;
        skill_category_map[skill] = std::vector< std::pair<size_t, double> >(m);
        for (auto & p : skill_category_map[skill])
            myin>>p.first>>p.second;
    }
    myin.close();
}

std::vector<size_t> company_category_map;
void read_company_category()
{
    std::ifstream myin(company_category_path);
    assert(myin);
    int n;
    myin >> n;
    company_category_map.resize(n);
    for (int i=0 ;i<n; i++)
        myin>>company_category_map[i];
    myin.close();
}

//  对不同的分类数量进行实验
//  公司分类数量20 40 60 80 100
//  技能分类数量20 40 60 80 100
void func20181206_train()
{

    std::cout<<"train log 20181206"<<std::endl;
    std::cout<<"model4 lamnda 1 1 5"<<std::endl;
    std::cout<<"different category num of company and skill"<<std::endl;

    wjy::sparse_tensor<double ,4> train_tensor;
    std::vector<int> mini_batch_nums = {50, 500, 1000, 1000, 1000};

    int years = 5;

    for (int i=20; i<=100; i+=20)
    {
        company_category_path = "../data/20180906/" + std::to_string(i) + "_category_of_company.txt";
        read_company_category();
        for (int j=20; j<=100; j+=20)
        {
            skill_category_path = "../data/20180906/skill_category_" + std::to_string(j) +".txt";
            std::string model_path = "../model/20181206_lambda115_c"+std::to_string(i)+"_s"+std::to_string(j)+"_";
            read_skill_category();

            std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
            std::cout<<"company category num:"<<i<<std::endl;
            std::cout<<"skill category num:"<<j<<std::endl;


            wjy::load_sparse_tensor(train_tensor, train_tensor_path);
            auto train_tensors  = wjy::split_sparse_tensor(train_tensor, 3);

            std::vector< std::vector<double> > old_parameters;

            for (int k=0; k<years; k++)
            {
                train_tensors[k][{24530, 11825, 680}] = 0;
                wjy::my_model_4< double, 3 > pred(
                    std::move(train_tensors[k]), 10, 0.5, mini_batch_nums[k], 
                    1/*0*/, company_category_map, 
                    1/*0*/, old_parameters,
                    5/*0*/, skill_category_map
                );
                pred.train(sgd, std::cout, distribution1);
                pred.train(gd, std::cout);
                old_parameters.push_back( std::move(pred.get_parameters()) );
                pred.save(model_path + std::to_string(k+2013)+".mod");
            }
        }
    }

}

void func20181206_evaluate()
{
    wjy::sparse_tensor<double ,4> test_tensor;
    wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    auto test_tensors = wjy::split_sparse_tensor(test_tensor, 3);

    int years = 5;

    for (int i=20; i<=100; i+=20)
    {
        for (int j=20; j<=100; j+=20)
        {
            std::string model_path = "../model/20181206_lambda115_c"+std::to_string(i)+"_s"+std::to_string(j)+"_";

            std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
            std::cout<<"company category num:"<<i<<std::endl;
            std::cout<<"skill category num:"<<j<<std::endl;

            //所有年合在一块评估
            std::vector< wjy::predictor<double, wjy::sparse_tensor_index<3> > * > preds;//有内存泄露 懒得delete了
            for (int k=0; k<years; k++) preds.push_back( new wjy::my_model_4<double, 3>() );

            for (int k=0; k<years; k++) preds[k]->load(model_path + std::to_string(k+2013)+".mod");

            auto ae = all_evaluations_1(test_tensors, preds);

            for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";
        }
    }
}

void func20181221_train()
{
    std::cout<<"train log 20181221"<<std::endl;
    std::cout<<"all lamnda 0.5 1 1 5"<<std::endl;
    std::cout<<"different k"<<std::endl;

    wjy::sparse_tensor<double ,4> train_tensor;
    std::vector< wjy::sparse_tensor<double, 3> > train_tensors;
    std::vector<int> mini_batch_nums = {50, 500, 1000, 1000, 1000};
    std::vector< std::vector<double> > old_parameters;

    company_category_path   = "../data/20180906/20_category_of_company.txt";  
    skill_category_path     = "../data/20180906/skill_category_40.txt";
    read_company_category();
    read_skill_category();

    int years=5;

    auto factory = [&](int i, std::size_t k, int year)-> wjy::predictor<double, wjy::sparse_tensor_index<3> > *{
        switch(i){
            case 0:
                return new wjy::tucker_decomposition<double, 3>(std::move(train_tensors[year]), {k, k, k}, 0.5, mini_batch_nums[year]);
            case 1:
                return new wjy::canonical_decomposition<double, 3>(std::move(train_tensors[year]), k, 0.5, mini_batch_nums[year]);
            case 2:
                return new wjy::pairwise_interaction_tensor_factorization<double, 3>(std::move(train_tensors[year]), k, 0.5, mini_batch_nums[year]);
            case 3:
                return new wjy::my_model_4<double, 3>(std::move(train_tensors[year]), k, 0.5, mini_batch_nums[year], 
                    1/*0*/, company_category_map, 
                    1/*0*/, old_parameters,
                    5/*0*/, skill_category_map);
            default:
                return nullptr;
        }
    };

    wjy::predictor< double, wjy::sparse_tensor_index<3> > * pred = nullptr;

    for (int i=0; i<4; i++)
    {
        for (int k=5; k<=30; k+=5)
        {
            wjy::load_sparse_tensor(train_tensor, train_tensor_path);
            train_tensors = wjy::split_sparse_tensor(train_tensor, 3);
            old_parameters.clear();

            std::string model_path = "../model/20181221_lambda115_i"+std::to_string(i)+"_k"+std::to_string(k)+"_";
            std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
            std::cout<<"i= "<<i<<std::endl;
            std::cout<<"k= "<<k<<std::endl;
        

            for (int year=0; year<years; year++)
            {
                if (pred != nullptr) delete pred;
                pred = factory(i, k, year);
                assert(pred!=nullptr);

                pred->train(sgd, std::cout, distribution1);
                pred->train(gd, std::cout);
                old_parameters.push_back( std::move(pred->get_parameters()) );
                pred->save(model_path + std::to_string(k+2013)+".mod");
            }
        }
    }
}

void func20181221_evaluate()
{
    wjy::sparse_tensor<double ,4> test_tensor;
    wjy::load_sparse_tensor(test_tensor, test_tensor_path);
    auto test_tensors = wjy::split_sparse_tensor(test_tensor, 3);

    int years = 5;

    auto factory = [](int i)-> wjy::predictor<double, wjy::sparse_tensor_index<3> > *{
        switch(i){
            case 0:
                return new wjy::tucker_decomposition<double, 3>();
            case 1:
                return new wjy::canonical_decomposition<double, 3>();
            case 2:
                return new wjy::pairwise_interaction_tensor_factorization<double, 3>();
            case 3:
                return new wjy::my_model_4<double, 3>();
            default:
                return nullptr;
        }
    };


    for (int i=0; i<4; i++)
    {
        for (int k=5; k<=30; k+=5)
        {
            std::string model_path = "../model/20181221_lambda115_i"+std::to_string(i)+"_k"+std::to_string(k)+"_";

            std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
            std::cout<<"i= "<<i<<std::endl;
            std::cout<<"k= "<<k<<std::endl;

            //所有年合在一块评估
            std::vector< wjy::predictor<double, wjy::sparse_tensor_index<3> > * > preds;//有内存泄露 懒得delete了
            for (int year=0; year<years; year++) preds.push_back( factory(i) );

            for (int year=0; year<years; year++) preds[year]->load(model_path + std::to_string(k+2013)+".mod");

            auto ae = all_evaluations_1(test_tensors, preds);

            for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";
        }
    }
}



int main(int args, const char* argv[])
{

    // wjy::sparse_tensor<double ,4> train_tensor, test_tensor, tensor;
    // wjy::load_sparse_tensor(tensor, tensor_path);
    // wjy::load_sparse_tensor(train_tensor, train_tensor_path);
    // wjy::load_sparse_tensor(test_tensor, test_tensor_path);

    // read_skill_category();
    // read_company_category();

    // auto train_tensors  = wjy::split_sparse_tensor(train_tensor, 3);
    // auto test_tensors   = wjy::split_sparse_tensor(test_tensor, 3);
    // std::vector< std::vector<double> > old_parameters;

    // std::string file_path = "../model/20181013_2_";
    // int n = 5;    //  2013-2017 5 years
    
    // std::vector<int> mini_batch_nums = {50, 500, 1000, 1000, 1000};

    // std::cout<<"PITF + 1, 1 0 0"<<std::endl;
    // for (int i=0; i<n; i++)
    // {
    //     train_tensors[i][{24530, 11825, 680}] = 0;
    //     wjy::my_model_4< double, 3 > pred(
    //         std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i], 
    //         1/*0*/, company_category_map, 
    //         0/*1*/, old_parameters,
    //         0/*5*/, skill_category_map
    //     );
    //     // wjy::my_model_2<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i], old_parameters, 2);
    //     // wjy::pairwise_interaction_tensor_factorization<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, mini_batch_nums[i]);
    //     //wjy::my_model_1<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i]);
    //     // wjy::my_model_3<double, 3> pred(std::move(train_tensors[i]), 10, 0.5, category, 0.2, mini_batch_nums[i], old_parameters, 0.1);
    //     pred.train(sgd, std::cout, distribution1);
    //     pred.train(gd, std::cout);
    //     old_parameters.push_back( std::move(pred.get_parameters()) );
    //     pred.save(file_path + std::to_string(i+2013)+".mod");
    // }

    // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_4< double, 3 >();
    // // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_2<double, 3>();
    // // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::pairwise_interaction_tensor_factorization<double, 3>();
    // //wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_1<double, 3>();
    // //年份分开计算各种指标
    // // wjy::tensor_factorization_predictor<double, 3> * pred = new wjy::my_model_3<double, 3>();
    // for (int i=0; i<n; i++)
    // {
    //     pred->load(file_path + std::to_string(i+2013)+".mod");
    //     auto ae = all_evaluations(test_tensors[i], *pred);
    //     for (auto p : ae) std::cout<<p.first<<" "<<p.second<<"\n";
    // }


    /*
        "百度", "c++研发工程师",
        "美团点评",
        "腾讯".
        "美团网",
        "饿了么",
        "京东",
        "阿里巴巴",
    */

    // for (auto & a : print_top_k_company(20)) wjy::println(a);
    // for (auto & a : print_top_k_position_of_company("百度", 20)) wjy::println(a);
    // for (int i=0; i<n; i++)
    // {
    //     pred->load(file_path + std::to_string(i+2013)+".mod");
    //     wjy::println( top_k_skill_of_company_position(*pred, "百度", "c++研发工程师", 20) );
    // }

    // print_companies_skills_count({"百度", "京东", "腾讯", "阿里巴巴"});

    //func20181206_train();
    //func20181206_evaluate();


    func20181221_train();
    //func20181221_evaluate();
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


