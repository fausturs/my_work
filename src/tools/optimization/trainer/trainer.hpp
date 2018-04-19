//
//  Trainer.hpp
//  cpp_test
//
//  Created by wjy on 2018/4/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef trainer_hpp
#define trainer_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#include "tools.hpp"
#include "Timer.hpp"

namespace wjy {

    template<typename T>
    class trainer{
    protected:
        std::vector<T> parameters;
        //  parameters += g*ratio;
        virtual void update_parameters( const std::vector<T> &g, T ratio);
    public:
        trainer() = default;
        trainer(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition);
        //  movable and copiable
        //  if you want your derived class can be moved, these functions should be write in derived class too.
        //  because the destroy must be virtual and be declared.
        //  compiler will not generate the move operator by default, unless you declare it. maybe, it's a bug.
        trainer(trainer&&) = default;
        trainer& operator=(trainer&&) = default;
        trainer(const trainer&) = default;
        trainer& operator=(const trainer&) = default;
        
        //  these functions and values should be set before training.
        //  "gradient", "hessian" and "mini_batch_stochastic_gradient" three functions sometimes can be nullptr.
        //  it depends on the define of member function "train".
        size_t parameters_num, mini_batch_num;
        std::function< T(const std::vector<T>&) > loss;
        std::function< std::vector<T>(const std::vector<T>&) > gradient;
        std::function< std::vector<T>(const std::vector<T>&) > hessian;
        std::function< std::vector<T>(const std::vector<T>&, size_t) > mini_batch_stochastic_gradient;
        std::function< std::vector<T>(const std::vector<T>&) > proximal;
        //
        size_t epoch_size, iterate_num;
        T init_learning_rate, convergence_condition;
        
        virtual void train();
        virtual void train(std::ostream& log) = 0;
        virtual void init_parameters(std::function< T(void) > distribution);
        virtual void init_parameters(const std::string &file_path);
        virtual void init_parameters(const std::vector<T> &para);
        //  return a copy of parameters;
        virtual std::vector< T > get_parameters() const;
        virtual void clear();
        virtual void test();
        
        virtual ~trainer() = default;
    };
    
    template< typename T>
    trainer<T>::trainer(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition)
    {
        this->iterate_num = iterate_num;
        this->epoch_size = epoch_size;
        this->init_learning_rate = init_learning_rate;
        this->convergence_condition = convergence_condition;
    }
    
    template< typename T>
    void trainer<T>::update_parameters( const std::vector<T> &g, T ratio)
    {
        for (size_t i=0; i<parameters.size(); i++)
            parameters[i] += g[i]*ratio;
    }
    
    template< typename T>
    void trainer<T>::train()
    {
        std::ostringstream myout;
        train(myout);
    }
    
    template< typename T>
    void trainer<T>::init_parameters(std::function< T() > distribution)
    {
        parameters.resize( parameters_num );
        std::generate_n(parameters.begin(), parameters_num, distribution);
    }
    
    template< typename T>
    void trainer<T>::init_parameters(const std::string &file_path)
    {
        std::ifstream myin(file_path);
        assert(myin);
        parameters.resize( parameters_num );
        T element;
        size_t i = 0;
        while (myin>>element) parameters[i++] = element;
    }
    
    template< typename T>
    void trainer<T>::init_parameters(const std::vector<T> &para)
    {
        parameters.resize( para.size() );
        std::copy(para.begin(), para.end(), parameters.begin());
    }
    
    template< typename T>
    std::vector<T> trainer<T>::get_parameters() const
    {
        return parameters;
    }
    
    template< typename T>
    void trainer<T>::test()
    {
        std::cout<<"This is class trainer."<<std::endl;
    }
    
    template< typename T>
    void trainer<T>::clear()
    {
        parameters.clear();
        parameters_num = mini_batch_num = epoch_size = iterate_num = 0;
        loss = nullptr;
        gradient = nullptr;
        hessian = nullptr;
        mini_batch_stochastic_gradient = nullptr;
        proximal = nullptr;
        init_learning_rate = convergence_condition = static_cast<T>(0);
    }
    
}


#endif /* Trainer_hpp */
