//
//  predictor.hpp
//  cpp_test
//
//  Created by wjy on 2018/4/17.
//  Copyright © 2018年 wjy. All rights reserved.
//

#ifndef predicter_hpp
#define predicter_hpp

#include <iostream>
#include <iterator>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "trainer.hpp"

namespace wjy{
    //  predictor like a function combine parameters and data.
    //  parameters' type is std::vector<T>
    //  when we need to make prediction to a single data, the single data input type is V
    template < typename T, typename V >
    class predictor{
    protected:
        virtual std::vector<T> gradient(const std::vector<T>&) const;
        virtual std::vector<T> hessian(const std::vector<T>&) const;
        virtual std::vector<T> mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const;
        virtual std::vector<T> proximal(const std::vector<T>&) const;
        virtual T loss(const std::vector<T>&) const = 0;
        virtual T _predict(const std::vector<T>&, const V&)const = 0;
        
        virtual void deploy_traner(std::shared_ptr< trainer<T> > my_trainer);
        
        std::vector<T> parameters;
        size_t parameters_num, mini_batch_num;
        
    public:
        
        predictor() = default;
        explicit predictor(size_t mini_batch_num);
        //  movable
        //  if you want your derived class can be moved, these functions should be write in derived class too.
        //  because the destroy must be virtual and be declared.
        //  compiler will not generate the move operator by default, unless you declare it. maybe, it's a bug.
        predictor(predictor&&) = default;
        predictor& operator=(predictor&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete
        
        virtual T predict(const V&)const;

        
        //  trainer, training data and other settings
        virtual void train(std::shared_ptr< trainer<T> >, std::function< T()> distribution=nullptr);
        virtual void train(std::shared_ptr< trainer<T> >, std::ostream& log, std::function< T()> distribution=nullptr);
        virtual void train(std::shared_ptr< trainer<T> >, const std::vector<T>& v);
        virtual void train(std::shared_ptr< trainer<T> >, std::ostream& log, const std::vector<T>& v);
        
        virtual std::vector<T> get_parameters() const;
        virtual void clear();
        virtual void save_parameters(std::ostream&) const;
        virtual void load_parameters(std::istream&);
        virtual void test();
        
        virtual ~predictor() = default;
    };
    
    template < typename T, typename V >
    std::vector<T> predictor<T, V>::gradient(const std::vector<T>&) const {return {};}
    template < typename T, typename V >
    std::vector<T> predictor<T, V>::hessian(const std::vector<T>&) const {return {};}
    template < typename T, typename V >
    std::vector<T> predictor<T, V>::mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const {return {};}
    template < typename T, typename V >
    std::vector<T> predictor<T, V>::proximal(const std::vector<T>&) const {return {};}
    
    template < typename T, typename V >
    void predictor<T, V>::deploy_traner(std::shared_ptr< trainer<T> > my_trainer)
    {
        if (my_trainer==nullptr) return;
        my_trainer->mini_batch_num = this->mini_batch_num;
        my_trainer->parameters_num = this->parameters_num;
        using namespace std::placeholders;
        my_trainer->loss = std::bind(std::mem_fn(&predictor<T, V>::loss), this, _1);
        my_trainer->gradient = std::bind(std::mem_fn(&predictor<T, V>::gradient), this, _1);
        my_trainer->hessian = std::bind(std::mem_fn(&predictor<T, V>::hessian), this, _1);
        my_trainer->mini_batch_stochastic_gradient = std::bind(std::mem_fn(&predictor<T, V>::mini_batch_stochastic_gradient), this, _1, _2);
        my_trainer->proximal = std::bind(std::mem_fn(&predictor<T, V>::proximal), this, _1);
        
        my_trainer->init_parameters(parameters);
    }
    
    template < typename T, typename V >
    predictor<T, V>::predictor(size_t mini_batch_num)
    :mini_batch_num(mini_batch_num), parameters_num(0)
    {
    }
    
    template < typename T, typename V >
    T predictor<T, V>::predict(const V& x) const
    {
        return _predict(parameters, x);
    }
    
    template < typename T, typename V >
    void predictor<T, V>::train(std::shared_ptr< trainer<T> > my_trainer, std::function< T() > distribution)
    {
        std::ostringstream myout;
        train(my_trainer, myout, distribution);
    }
    
    template < typename T, typename V >
    void predictor<T, V>::train(std::shared_ptr< trainer<T> > my_trainer, std::ostream& log, std::function< T() > distribution)
    {
        if (my_trainer == nullptr) return;
        deploy_traner(my_trainer);
        if (distribution!=nullptr)
            my_trainer->init_parameters(distribution);
        else
            my_trainer->init_parameters(parameters);
        my_trainer->train(log);
        parameters = my_trainer->get_parameters();
    }
    
    template < typename T, typename V >
    void predictor<T, V>::train(std::shared_ptr< trainer<T> > my_trainer, const std::vector<T>& v)
    {
        std::ostringstream myout;
        train(my_trainer, myout, v);
    }
    
    template < typename T, typename V >
    void predictor<T, V>::train(std::shared_ptr< trainer<T> > my_trainer, std::ostream& log, const std::vector<T>& v)
    {
        if (my_trainer == nullptr) return;
        deploy_traner(my_trainer);
        my_trainer->init_parameters(v);
        my_trainer->train(log);
        parameters = my_trainer->get_parameters();
    }
    
    template < typename T, typename V >
    std::vector<T> predictor<T, V>::get_parameters() const
    {
        return parameters;
    }
    
    template < typename T, typename V >
    void predictor<T, V>::clear()
    {
        parameters_num = mini_batch_num = 0;
    }
    
    template < typename T, typename V >
    void predictor<T, V>::save_parameters(std::ostream& myout) const
    {
        assert(myout);
        myout<<parameters_num<<" "<<mini_batch_num<<" ";
        std::copy(parameters.begin(), parameters.end(), std::ostream_iterator<T>(myout, " "));
        myout<<std::endl;
    }
    
    template < typename T, typename V >
    void predictor<T, V>::load_parameters(std::istream& myin)
    {
        assert(myin);
        myin>>parameters_num>>mini_batch_num;
        parameters.resize(parameters_num);
        std::copy_n(std::istream_iterator<T>(myin), parameters_num, parameters.begin());
    }
    
    template < typename T, typename V >
    void predictor<T, V>::test()
    {
        std::cout<<"This is class predictor."<<std::endl;
    }
}


#endif /* predicter_hpp */
