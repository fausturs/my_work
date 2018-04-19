#ifndef canonical_decomposition_hpp
#define canonical_decomposition_hpp

#include "tensor_factorization_predictor.hpp"

namespace wjy {
    //  \min \|A - p_1 \times ... \times p_{k}\|_F + \lambda(\sum_i \|p_i\|_F)
    template<typename T, size_t kth_order>
    class canonical_decomposition final: public tensor_factorization_predictor<T, kth_order >{
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
        
        //  coefficient of norms
        T lambda;
        
        std::vector<T> n_vector_product(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index, const std::unordered_set<size_t>& except) const;
    public:
        canonical_decomposition() = default;
        canonical_decomposition(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num);
        //  movable
        canonical_decomposition(canonical_decomposition&&) = default;
        canonical_decomposition& operator=(canonical_decomposition&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete
        
        virtual void test() override;
        
        virtual ~canonical_decomposition()override = default;
    };
    
    template<typename T, size_t kth_order>
    std::vector<T> canonical_decomposition<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        std::vector<T> g(parameters_num, 0);
        //  F-norm's gradient, it equals to \lambda(\|v\|^2)' = 2\lambda v
        vector_sum(g.begin(), g.end(), para.begin(), 2*lambda);
        //  gradient of \sum (A - p_1 \times ... \times p_k)^2
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        size_t vector_length = matrixes_size.front().second;
        for (const auto & entry : tensor_A)
        {
            auto & index = entry.first;
            auto & value = entry.second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            for (size_t i=0; i<kth_order; i++)
            {
                auto g_v = n_vector_product(para, offset, index, {i});
                auto temp_it = g.begin() + offset[i] + index[i]*vector_length;
                vector_sum(temp_it, temp_it+vector_length, g_v.begin(), temp);
            }
        }
        return g;
    }
    
    template<typename T, size_t kth_order>
    std::vector<T> canonical_decomposition<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i) const
    {
        std::vector<T> g(parameters_num, 0);
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        size_t vector_length = matrixes_size.front().second;
        for (auto it = batchs[batch_i]; it!=batchs[batch_i+1]; it++)
        {
            auto & index = it->first;
            auto & value = it->second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            for (size_t i=0; i<kth_order; i++)
            {
                auto g_v = n_vector_product(para, offset, index, {i});
                auto offset_i = offset[i]+index[i]*vector_length;
                auto temp_it = g.begin()+ offset_i;
                //  gradient of square error
                vector_sum(temp_it, temp_it+vector_length, g_v.begin(), temp);
                //  gradient of F-norm
                vector_sum(temp_it, temp_it+vector_length, para.begin()+offset_i, 2*lambda/(counter.at(offset_i)));
            }
        }
        return g;
    }
    
    template<typename T, size_t kth_order>
    T canonical_decomposition<T, kth_order>::loss(const std::vector<T>& para) const
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
    
    template<typename T, size_t kth_order>
    T canonical_decomposition<T, kth_order>::_predict(const std::vector<T>& para, const sparse_tensor_index<kth_order>& index) const
    {
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*matrixes_size[i-1].second;
        auto ans = n_vector_product(para, offset, index, {});
        return std::accumulate(ans.begin(), ans.end(), static_cast<T>(0));
    }
    
    template<typename T, size_t kth_order>
    canonical_decomposition<T, kth_order>::canonical_decomposition(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num)
    :base_class(std::move(tensor_A), a_fill_array<size_t, kth_order>(parameters_rank), mini_batch_num)
    ,lambda(lambda)
    {
        //  empty here
        //  template function "a_fill_array" return a array fill with the same element,
        //  which defined in tools.hpp
    }
    
    template< typename T, size_t kth_order >
    void canonical_decomposition<T, kth_order>::test()
    {
        std::cout<<"This is class canonical_decomposition."<<std::endl;
    }
    
    template< typename T, size_t kth_order >
    std::vector<T> canonical_decomposition<T, kth_order>::n_vector_product(const std::vector<T>& para, const std::vector< size_t >& offset, const sparse_tensor_index<kth_order>& index, const std::unordered_set<size_t>& except) const
    {
        //  in this model, every vector need to be the same length
        size_t vector_lenght = matrixes_size.front().second;
        std::vector<T> ans(vector_lenght, 1);
        for (size_t i=0; i<kth_order; i++)
        {
            if (except.count(i)!=0) continue;
            auto para_it = para.begin() + offset[i] + index[i]*vector_lenght;
            auto ans_it = ans.begin();
            for (size_t j=0; j<vector_lenght; j++)
                *(ans_it++) *= *(para_it++);
        }
        return ans;
    }
}

#endif
