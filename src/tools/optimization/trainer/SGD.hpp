//
//  SGD.hpp
//  cpp_test
//
//  Created by wjy on 2018/4/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef SGD_hpp
#define SGD_hpp

#include "trainer.hpp"

namespace wjy {
    template<typename T>
    class SGD final : public trainer<T>{
    private:
        using trainer<T>::parameters;
        using trainer<T>::update_parameters;
        
        int random_seed;
    public:
        SGD() = delete;
        SGD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition, int random_seed=0);
        //  movable and copiable
        SGD(SGD&&) = default;
        SGD& operator=(SGD&&) = default;
        SGD(const SGD&) = default;
        SGD& operator=(const SGD&) = default;
        
        using trainer<T>::loss;
        using trainer<T>::gradient;
        using trainer<T>::mini_batch_stochastic_gradient;
        using trainer<T>::epoch_size;
        using trainer<T>::mini_batch_num;
        using trainer<T>::iterate_num;
        using trainer<T>::init_learning_rate;
        using trainer<T>::convergence_condition;
        
        virtual void train(std::ostream& log) override;
        virtual void test() override;
        virtual ~SGD() override = default ;
    };
    
    template <typename T>
    SGD<T>::SGD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition, int random_seed)
    :trainer<T>(iterate_num, epoch_size, init_learning_rate, convergence_condition)
    {
        this->random_seed = random_seed;
    }
    
    template <typename T>
    void SGD<T>::train(std::ostream& log)
    {
        auto mt = (random_seed == -1)?std::mt19937( std::random_device{}() ):std::mt19937(random_seed);
        std::uniform_int_distribution<size_t> distribution(0, mini_batch_num-1);
        //
        log<<"training start..."<<std::endl;
        format_print<16>(log, "epoch", "loss", "s_g norm", "time(s)");
        wjy::Timer timer;
        timer.start();
        for (size_t iter=0; iter<iterate_num; iter++)
        {
            auto s_g = mini_batch_stochastic_gradient(parameters, distribution(mt));
            //  print log
            if (iter%epoch_size == 0)
            {
                T s_g_norm = 0;
                s_g_norm = std::sqrt(std::inner_product(s_g.begin(), s_g.end(), s_g.begin(), s_g_norm));
                T l = loss(parameters);
                timer.end();
                wjy::format_print<16>(log, iter/epoch_size, l, s_g_norm, timer.get_duration_s());
                //  convergence condition
                if (s_g_norm < convergence_condition) break;
            }
            //
            auto learning_rate = init_learning_rate;
            update_parameters(s_g, -learning_rate);
        }
        log<<"training finished!"<<std::endl;
        auto g = gradient(parameters);
        T g_norm = 0;
        g_norm = std::sqrt( std::inner_product(g.begin(), g.end(), g.begin(), g_norm) );
        log<<"gradient norm: "<<g_norm<<std::endl;
    }
    
    template <typename T>
    void SGD<T>::test()
    {
        std::cout<<"This is class SGD."<<std::endl;
    }
}


#endif /* SGD_hpp */
