//
//  GD.hpp
//  cpp_test
//
//  Created by wjy on 2018/4/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef GD_hpp
#define GD_hpp

#include "trainer.hpp"

namespace wjy {
    template <typename T>
    class GD final : public trainer<T>{
    protected:
        using trainer<T>::parameters;
        using trainer<T>::update_parameters;
    public:
        GD() = delete;
        GD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition);
        //  movable and copiable
        GD(GD&&) = default;
        GD& operator=(GD&&) = default;
        GD(const GD&) = default;
        GD& operator=(const GD&) = default;
        
        using trainer<T>::loss;
        using trainer<T>::gradient;
        using trainer<T>::epoch_size;
        using trainer<T>::iterate_num;
        using trainer<T>::init_learning_rate;
        using trainer<T>::convergence_condition;
        
        virtual void train(std::ostream& log) override;
        virtual void test() override;
        virtual ~GD()override = default;
    };
    
    template <typename T>
    GD<T>::GD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition)
    :trainer<T>(iterate_num, epoch_size, init_learning_rate, convergence_condition)
    {
    }
    
    template <typename T>
    void GD<T>::train(std::ostream& log)
    {
        log<<"training start..."<<std::endl;
        format_print<16>(log, "epoch", "loss", "g norm", "time(s)");
        wjy::Timer timer;
        timer.start();
        for (size_t iter=0; iter<iterate_num; iter++)
        {
            auto g = gradient(parameters);
            //  print log
            if ((iter%epoch_size) == 0)
            {
                T g_norm = 0;
                g_norm = std::inner_product(g.begin(), g.end(), g.begin(), g_norm);
                g_norm = std::sqrt(g_norm);
                T l = loss(parameters);
                timer.end();
                wjy::format_print<16>(log, iter/epoch_size, l, g_norm, timer.get_duration_s());
                //  convergence condition
                if (g_norm < convergence_condition) break;
            }
            //
            auto learning_rate = init_learning_rate;
            update_parameters(g, -learning_rate);
        }
        log<<"training finished!"<<std::endl;
        auto g = gradient(parameters);
        T g_norm = 0;
        g_norm = std::sqrt( std::inner_product(g.begin(), g.end(), g.begin(), g_norm) );
        log<<"gradient norm: "<<g_norm<<std::endl;
    }
    
    template <typename T>
    void GD<T>::test()
    {
        std::cout<<"This is class GD."<<std::endl;
    }
}



#endif /* GD.hpp */
