#ifndef tucker_decomposition_hpp
#define tucker_decomposition_hpp

#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <cassert>

#include "tools.hpp"
#include "sparse_tensor.hpp"
#include "tensor_factorization_predictor.hpp"

namespace wjy {
    //  \min \|A - s \times p_1 \times ... \times p_{k}\|_F + \lambda(\sum_i \|p_i\|_F + \|s\|_F)
    template< typename T, size_t kth_order >
    class tucker_decomposition final: public tensor_factorization_predictor<T, kth_order >{
        using base_class = tensor_factorization_predictor<T, kth_order >;
        using base_class::tensor_A;
        using base_class::matrixes_size;
        using base_class::batchs;
        using base_class::counter;
        using base_class::parameters;
        using base_class::parameters_num;
        
        virtual std::vector<T> gradient(const std::vector<T>&) const override;
        virtual std::vector<T> mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const override;
        virtual T loss(const std::vector<T>&) const override;
        virtual T _predict(const std::vector<T>&, const sparse_tensor_index<kth_order>&) const override;
        
        size_t tensor_s_size;
        //  coefficient of norms
        T lambda;
        
        //
        std::vector< T > tensor_s_multiply_n_vector(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index, const std::unordered_set<size_t>& except) const;
        std::vector< T > calculate_s_gradient_at(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index) const;
    public:
        tucker_decomposition() = default;
        tucker_decomposition(sparse_tensor<T, kth_order>&& tensor_A, const std::array<size_t, kth_order>& parameters_rank, T lambda, size_t mini_batch_num);
        //  movable
        tucker_decomposition(tucker_decomposition&&) = default;
        tucker_decomposition& operator=(tucker_decomposition&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete
        
        virtual void clear() override;
        virtual void save_parameters(std::ostream&) const override;
        virtual void load_parameters(std::istream&) override;
        virtual void test() override;
        
        virtual ~tucker_decomposition()override = default;
    };
    
    template< typename T, size_t kth_order >
    std::vector<T> tucker_decomposition<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        std::vector<T> g(parameters_num, 0);
        //  F-norm's gradient, it equals to \lambda(\|v\|^2)' = 2\lambda v
        vector_sum(g.begin(), g.end(), para.begin(), 2*lambda);
        //  gradient of \sum (A - s*para)^2
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        for (const auto & entry : tensor_A)
        {
            auto & index = entry.first;
            auto & value = entry.second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            for (size_t i=0; i<kth_order; i++)
            {
                auto g_v = tensor_s_multiply_n_vector(para, offset, index, {i});
                auto temp_it = g.begin()+offset[i]+index[i]*matrixes_size[i].second;
                vector_sum(temp_it, temp_it+matrixes_size[i].second, g_v.begin(), temp);
            }
            auto g_s = calculate_s_gradient_at(para, offset, index);
            vector_sum(g.begin()+offset.back(), g.end(), g_s.begin(), temp );
        }
        return g;
    }
    template< typename T, size_t kth_order >
    std::vector<T> tucker_decomposition<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i) const
    {
        std::vector<T> g(parameters_num, 0);
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        //
        size_t batch_size = 0;
        for (auto it = batchs[batch_i]; it!=batchs[batch_i+1]; it++,batch_size++)
        {
            auto & index = it->first;
            auto & value = it->second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            for (size_t i=0; i<kth_order; i++)
            {
                auto g_v = tensor_s_multiply_n_vector(para, offset, index, {i});
                auto offset_i = offset[i]+index[i]*matrixes_size[i].second;
                auto temp_it = g.begin()+ offset_i;
                //  gradient of square error
                vector_sum(temp_it, temp_it+matrixes_size[i].second, g_v.begin(), temp);
                //  gradient of F-norm
                vector_sum(temp_it, temp_it+matrixes_size[i].second, para.begin()+offset_i, 2*lambda/(counter.at(offset_i)));
            }
            auto g_s = calculate_s_gradient_at(para, offset, index);
            vector_sum(g.begin()+offset.back(), g.end(), g_s.begin(), temp );
        }
        //  tensor_s's F-norm gradient
        T ratio = 2*lambda*batch_size/(tensor_A.size());
        vector_sum(g.begin()+offset.back(), g.end(), para.begin()+offset.back(), ratio);
        return g;
    }
    
    template< typename T, size_t kth_order >
    T tucker_decomposition<T, kth_order>::loss(const std::vector<T>& para) const
    {
        T l = 0;
        //  all of the F-norm
        l = lambda * std::inner_product(para.begin(), para.end(), para.begin(), l);
        //  square error
        for (auto & entry : tensor_A)
        {
            auto & index = entry.first;
            auto & value = entry.second;
            auto pred = _predict(para, index);
            l += (pred - value)*(pred - value);
        }
        return l;
    }
    
    template< typename T, size_t kth_order >
    tucker_decomposition<T, kth_order>::tucker_decomposition(sparse_tensor<T, kth_order>&& tensor_A, const std::array<size_t, kth_order>& parameters_rank, T lambda, size_t mini_batch_num)
    :base_class(std::move(tensor_A), parameters_rank, mini_batch_num), lambda(lambda)
    {
        //  update parameters_num
        tensor_s_size = 1;
        for (auto & size : matrixes_size) tensor_s_size *= size.second;
        parameters_num += tensor_s_size;
    }
    
    template< typename T, size_t kth_order >
    T tucker_decomposition<T, kth_order>::_predict(const std::vector<T>& para, const sparse_tensor_index<kth_order>& index) const
    {
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        auto ans = tensor_s_multiply_n_vector(para, offset, index, {});
        return ans[0];
    }
    
    
    template< typename T, size_t kth_order >
    void tucker_decomposition<T, kth_order>::clear()
    {
        tensor_s_size = 0;
        lambda = 0;
        base_class::clear();
    }
    template< typename T, size_t kth_order >
    void tucker_decomposition<T, kth_order>::save_parameters(std::ostream& myout) const
    {
        base_class::save_parameters(myout);
        myout<<tensor_s_size<<std::endl;
    }
    template< typename T, size_t kth_order >
    void tucker_decomposition<T, kth_order>::load_parameters(std::istream& myin)
    {
        base_class::load_parameters(myin);
        myin>>tensor_s_size;
    }
    
    template< typename T, size_t kth_order >
    void tucker_decomposition<T, kth_order>::test()
    {
        std::cout<<"This is class tucker_decomposition."<<std::endl;
    }
    
    //
    template< typename T, size_t kth_order >
    std::vector< T > tucker_decomposition<T, kth_order>::tensor_s_multiply_n_vector(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index, const std::unordered_set<size_t>& except) const
    {
        std::vector< size_t > dims(kth_order);
        for (size_t i=0; i<kth_order; i++) dims[i] = matrixes_size[i].second;
        auto tensor_s_it = para.begin() + offset.back();
        auto dims_rit = dims.rbegin();
        std::vector< T > t1( tensor_s_it, tensor_s_it+tensor_s_size );
        std::vector< T > t2( t1.size() );
        for (size_t i=0; i<kth_order; i++,dims_rit++)
        {
            if (except.count(kth_order-i-1)) continue;
            auto v_it = para.begin() + offset[kth_order-i-1] + index[kth_order-i-1]*matrixes_size[kth_order-i-1].second;
            //  defined in tools.hpp
            tensor_multiply_vector<T>(t1.begin(), kth_order, dims.begin(), kth_order-i-1, v_it, t2.begin());
            std::swap(t1, t2);
            *dims_rit = 1;
        }
        size_t length = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});
        t1.resize(length);
        return t1;
    }
    
    template< typename T, size_t kth_order >
    std::vector< T > tucker_decomposition<T, kth_order>::calculate_s_gradient_at(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index) const
    {
        std::vector< T > g_s( tensor_s_size, 1);
        size_t pre_size = 1, dim_i = 1;
        size_t suf_size = tensor_s_size;
        for (size_t i=0; i<kth_order; i++)
        {
            dim_i = matrixes_size[i].second;
            suf_size /= dim_i;
            pre_size = tensor_s_size / (suf_size * dim_i);
            auto v_begin = para.begin() + offset[i] + index[i]*matrixes_size[i].second;
            for (size_t pre=0; pre<pre_size; pre++)
                for (size_t k=0; k<dim_i; k++)
                    for (size_t suf=0; suf<suf_size; suf++)
                    {
                        g_s[pre*(suf_size*dim_i)+k*(suf_size)+suf] *= (*(v_begin+k));
                    }
        }
        return g_s;
    }
}



#endif
