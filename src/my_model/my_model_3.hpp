#ifndef my_model_3_hpp
#define my_model_3_hpp

#include "my_model_1.hpp"

#include <algorithm>

namespace wjy{

    // 在my_model_1基础上加上一些以往的特征向量 

    template<typename T, size_t kth_order>
    class my_model_3 : public my_model_1<T, kth_order>{
        using base_class = my_model_1<T, kth_order >;
        using base_class::tensor_A;
        using base_class::parameters_num;
        using base_class::batchs;

        virtual std::vector<T> gradient(const std::vector<T>&) const override;
        virtual std::vector<T> mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const override;
        virtual T loss(const std::vector<T>&) const override;

        std::vector< std::vector<T> > old_parameters;
        T lambda_2;

        std::vector<T> gradient_of_old_parameters(const std::vector<T>&) const;
    public:
        my_model_3() = default;
        //  all vector in this model is the same length, so parameters_rank is a scalar.
        //  weights default = 1
        my_model_3(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1,  size_t mini_batch_num, std::vector< std::vector<T> > old_parameters, T lambda_2);
        //  init weights by a given array
        my_model_3(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1,  const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num, std::vector< std::vector<T> > old_parameters, T lambda_2);
            
        //  movable
        my_model_3(my_model_3&&) = default;
        my_model_3& operator=(my_model_3&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete 

        virtual void test() override;
        virtual ~my_model_3() override = default;
    };

    template<typename T, size_t kth_order>
    std::vector<T> my_model_3<T, kth_order>::gradient_of_old_parameters(const std::vector<T>& para) const
    {
        std::vector<T> temp_g(parameters_num, 0), temp(parameters_num, 0);
        size_t n = old_parameters.size();  
        for (size_t i=0; i<n; i++)
        {
            T ratio = lambda_2/(n-i);
            vector_sum(para.begin(), static_cast<T>(2), old_parameters[i].begin(), static_cast<T>(-2), parameters_num, temp.begin());
            vector_sum(temp_g.begin(), temp_g.end(), temp.begin(), ratio);
        }
        return temp_g;
    }

    template<typename T, size_t kth_order>
    std::vector<T> my_model_3<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        auto g = base_class::gradient(para);
        auto temp = gradient_of_old_parameters(para);

        vector_sum(g.begin(), g.end(), temp.begin(), static_cast<T>(1));
        return g;
    }

    template<typename T, size_t kth_order>
    std::vector<T> my_model_3<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i) const
    {
        auto g = base_class::mini_batch_stochastic_gradient(para, batch_i);
        auto temp = gradient_of_old_parameters(para);
        //  这一行的时间按复杂度是O(m)
        size_t m = std::distance(batchs[batch_i], batchs[batch_i+1]);
        vector_sum(g.begin(), g.end(), temp.begin(), static_cast<T>(m)/tensor_A.size());
        return g;
    }

    template<typename T, size_t kth_order>
    T my_model_3<T, kth_order>::loss(const std::vector<T>& para) const
    {
        T l = base_class::loss(para);
        std::vector<T> temp(parameters_num);
        size_t n = old_parameters.size();
        for (size_t i=0; i<n; i++)
        {
            //
            vector_sum(para.begin(), static_cast<T>(2), old_parameters[i].begin(), static_cast<T>(-2), parameters_num, temp.begin());
            T l_i = lambda_2 * std::inner_product(temp.begin(), temp.end(), temp.begin(), static_cast<T>(0));
            l_i /= n-i;
            l += l_i;
        }
        return l;
    }

    template<typename T, size_t kth_order>
    my_model_3<T, kth_order>::my_model_3(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1,  size_t mini_batch_num, std::vector< std::vector<T> > old_parameters, T lambda_2)
    :base_class(std::move(tensor_A), parameters_rank, lambda, category_map, lambda_1, mini_batch_num)
    ,lambda_2(lambda_2)
    {
        this->old_parameters = std::move(old_parameters);
    }

    template<typename T, size_t kth_order>
    my_model_3<T, kth_order>::my_model_3(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1,  const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num, std::vector< std::vector<T> > old_parameters, T lambda_2)
    :base_class(std::move(tensor_A), parameters_rank, lambda, category_map, lambda_1, mini_batch_num)
    ,lambda_2(lambda_2)
    {
        this->old_parameters = std::move(old_parameters);
    }

    template<typename T, size_t kth_order>
    void my_model_3<T, kth_order>::test()
    {
        std::cout<<"This is class my_model_3."<<std::endl;
    }
}
#endif