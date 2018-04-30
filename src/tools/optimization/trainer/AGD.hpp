#ifndef AGD_hpp
#define AGD_hpp

#include "trainer.hpp"

namespace wjy{
    //  Accelerated Gradient Descent, Nesterove 1983
    //  y_t = x_t + (1-\theta)(x_t - x_{t-1})
    //  x_{t+1} = y_t - learning_rate * gradient(y_t)
    template< typename T >
    class AGD final: public trainer<T>{ 
        using base_class = trainer<T>;
        using trainer<T>::parameters;
        using trainer<T>::update_parameters;

        std::vector<T> old_parameters, y;
        virtual void update_parameters( const std::vector<T> &g, T ratio) override;
    public:
        AGD() = delete;
        AGD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition, T theta);
        //  movable and copiable
        AGD(AGD&&) = default;
        AGD& operator=(AGD&&) = default;
        AGD(const AGD&) = default;
        AGD& operator=(const AGD&) = default;

        using trainer<T>::loss;
        using trainer<T>::gradient;
        using trainer<T>::epoch_size;
        using trainer<T>::iterate_num;
        using trainer<T>::init_learning_rate;
        using trainer<T>::convergence_condition;

        T theta;
        
        virtual void train(std::ostream& log) override;
        virtual void clear() override;
        virtual void test() override;
        virtual ~AGD() override = default;
    };

    template< typename T >
    AGD<T>::AGD(size_t iterate_num, size_t epoch_size, T init_learning_rate, T convergence_condition, T theta)
    :trainer<T>(iterate_num, epoch_size, init_learning_rate, convergence_condition)
    ,theta(theta)
    {
        //  empty here.
    }

    template <typename T>
    void AGD<T>::update_parameters( const std::vector<T> &g, T ratio)
    {
        std::swap(parameters, old_parameters);
        //  x_{t+1} = y - learning_rate f'(y)
        vector_sum(y.begin(), static_cast<T>(1), g.begin(), ratio, g.size(), parameters.begin());

    }

    template <typename T>
    void AGD<T>::train(std::ostream& log)
    {
        log<<"training start..."<<std::endl;
        format_print<16>(log, "epoch", "loss", "g norm", "time(s)");
        wjy::Timer timer;
        timer.start();
        old_parameters.resize(parameters.size());
        std::fill(old_parameters.begin(), old_parameters.end(), static_cast<T>(0));
        y.resize(parameters.size());
        for (size_t iter=0; iter<iterate_num; iter++)
        {
            //  y = x_t + (1-theta)(x_t - x_{t-1}) = (2-theta)x_t - (1-theta)x_{t-1}
            //  x_t is parameters, x_{t-1} is old_parameters;
            vector_sum(parameters.begin(), 2-theta, old_parameters.begin(), -(1-theta), parameters.size(), y.begin());
            //  f'(y) 
            auto g = gradient(y);
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
    void AGD<T>::clear()
    {
        base_class::clear();
        old_parameters.clear();
        y.clear();
    }

    template <typename T>
    void AGD<T>::test()
    {
        std::cout<<"This is class AGD."<<std::endl;
    }
}


#endif