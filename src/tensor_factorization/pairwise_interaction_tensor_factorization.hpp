#ifndef pairwise_interaction_tensor_factorization_hpp
#define pairwise_interaction_tensor_factorization_hpp

//  PITF (Steffen Rendle and Lars Schmidt-Thieme. 2010)
#include "tensor_factorization_predictor.hpp"

namespace wjy {
    //  let P^{i,j} denote a matrix parameter.
    //  predict({i_0,...,i_{k-1}}) = \sum_{p<q} w^{p, q} <P^{p, q}_{i_p}, P^{q, p}_{i_q}>
    //  and P^{p,*}'s size is same. so this class can use matrixes_size defined in base class
    //  only need update the value parameters_num
    
    template < typename T, size_t kth_order >
    class pairwise_interaction_tensor_factorization : public tensor_factorization_predictor<T, kth_order >{
        using base_class = tensor_factorization_predictor<T, kth_order >;
    protected:
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
        //  there are k(k-1) inner product in predirct function, each of them have a coefficient w^{p, q}.
        //  we store it as a vector rather than a matrix.
        //  all weights default = 1, unless it be seted.
        std::array<T, kth_order*(kth_order-1)/2> weights;
    public:
        pairwise_interaction_tensor_factorization() = default;
        //  all vector in this model is the same length, so parameters_rank is a scalar.
        //  weights default = 1
        pairwise_interaction_tensor_factorization(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num);
        //  init weights by a given array
        pairwise_interaction_tensor_factorization(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num);
        
        //  movable
        pairwise_interaction_tensor_factorization(pairwise_interaction_tensor_factorization&&) = default;
        pairwise_interaction_tensor_factorization& operator=(pairwise_interaction_tensor_factorization&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete
        
        virtual void clear() override;
        //  in fact, weights are not parameters.
        //  but we need it to predict, so we have to save/load it here
        virtual void save_parameters(std::ostream&) const override;
        virtual void load_parameters(std::istream&) override;
        virtual void test() override;
        
        virtual ~pairwise_interaction_tensor_factorization()override = default;
    };
    
    template < typename T, size_t kth_order >
    std::vector<T> pairwise_interaction_tensor_factorization<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        std::vector<T> g(parameters_num, 0);
        //  F-norm's gradient, it equals to \lambda(\|v\|^2)' = 2\lambda v
        vector_sum(g.begin(), g.end(), para.begin(), 2*lambda);
        //  gradient of square error
        size_t vector_length = matrixes_size.front().second;
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*vector_length*(kth_order-1);
        
        for (const auto & entry : tensor_A)
        {
            auto & index = entry.first;
            auto & value = entry.second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            size_t w_i = 0;
            for (size_t i=0; i<kth_order; i++)
                for (size_t j=i+1; j<kth_order; j++,w_i++)
                {
                    size_t offset_ij = offset[i] + matrixes_size[i].first*vector_length*(j-1) + index[i]*vector_length;
                    size_t offset_ji = offset[j] + matrixes_size[j].first*vector_length*(i  ) + index[j]*vector_length;
                    auto it_para_ij = para.begin() + offset_ij;
                    auto it_para_ji = para.begin() + offset_ji;
                    auto it_g_ij = g.begin() + offset_ij;
                    auto it_g_ji = g.begin() + offset_ji;
                    //  update gradient of square error
                    vector_sum(it_g_ij, it_g_ij+vector_length, it_para_ji, temp*weights[w_i]);
                    vector_sum(it_g_ji, it_g_ji+vector_length, it_para_ij, temp*weights[w_i]);
                }
        }
        return g;
    }
    template < typename T, size_t kth_order >
    std::vector<T> pairwise_interaction_tensor_factorization<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i) const
    {
        std::vector<T> g(parameters_num, 0);
        size_t vector_length = matrixes_size.front().second;
        std::vector< size_t > offset(kth_order+1, 0), old_offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
        {
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*vector_length*(kth_order-1);
            old_offset[i] = (i==0)?0:old_offset[i-1] + matrixes_size[i-1].first*vector_length;
        }

        for (auto it = batchs[batch_i]; it!=batchs[batch_i+1]; it++)
        {
            auto & index = it->first;
            auto & value = it->second;
            T pred = _predict(para, index), temp = (pred-value)*2;
            size_t w_i = 0;
            for (size_t i=0; i<kth_order; i++)
                for (size_t j=i+1; j<kth_order; j++,w_i++)
                {
                    //  update gradient of square error
                    size_t offset_ij = offset[i] + matrixes_size[i].first*vector_length*(j-1) + index[i]*vector_length;
                    size_t offset_ji = offset[j] + matrixes_size[j].first*vector_length*(i  ) + index[j]*vector_length;
                    auto it_para_ij = para.begin() + offset_ij;
                    auto it_para_ji = para.begin() + offset_ji;
                    auto it_g_ij = g.begin() + offset_ij;
                    auto it_g_ji = g.begin() + offset_ji;
                    vector_sum(it_g_ij, it_g_ij+vector_length, it_para_ji, temp*weights[w_i]);
                    vector_sum(it_g_ji, it_g_ji+vector_length, it_para_ij, temp*weights[w_i]);
                    //  update gradient of F-norm
                    size_t old_offset_i = old_offset[i] + index[i]*vector_length;
                    size_t old_offset_j = old_offset[j] + index[j]*vector_length;
                    vector_sum(it_g_ij, it_g_ij+vector_length, it_para_ij, 2*lambda/counter.at(old_offset_i));
                    vector_sum(it_g_ji, it_g_ji+vector_length, it_para_ji, 2*lambda/counter.at(old_offset_j));
                }
        }
        
        return g;
    }
    template < typename T, size_t kth_order >
    T pairwise_interaction_tensor_factorization<T, kth_order>::loss(const std::vector<T>& para) const
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
    template < typename T, size_t kth_order >
    T pairwise_interaction_tensor_factorization<T, kth_order>::_predict(const std::vector<T>& para, const sparse_tensor_index<kth_order>& index) const
    {
        T pred = 0;
        size_t w_i = 0;
        //  all vector is the same length
        size_t vector_length = matrixes_size.front().second;
        std::vector< size_t > offset(kth_order+1, 0);
        for (size_t i=0; i<=kth_order; i++)
            offset[i] = (i==0)?0:offset[i-1] + matrixes_size[i-1].first*vector_length*(kth_order-1);
        
        for (size_t i=0; i<kth_order; i++)
            for(size_t j=i+1; j<kth_order; j++,w_i++)
            {
                //  here use j-1 is because model only store k(k-1) matrixes.
                //  it means p^{i,i} is not in vector parameters.
                //  because j>i, so the offset of it1 should count only (j-1) matrixes.
                auto it1 = para.begin() + offset[i] + matrixes_size[i].first*vector_length*(j-1) + index[i]*vector_length;
                auto it2 = para.begin() + offset[j] + matrixes_size[j].first*vector_length*(i  ) + index[j]*vector_length;
                pred += weights[w_i] * std::inner_product(it1, it1+vector_length, it2, static_cast<T>(0));
            }
        return pred;
    }
    
    template < typename T, size_t kth_order >
    pairwise_interaction_tensor_factorization<T, kth_order>::pairwise_interaction_tensor_factorization(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, size_t mini_batch_num)
    :base_class(std::move(tensor_A), a_fill_array<size_t, kth_order>(parameters_rank), mini_batch_num)
    ,lambda(lambda)
    {
        //  set all weights = 1
        weights = a_fill_array<T, kth_order*(kth_order-1)/2>(1);
        //  update parameters_num;
        parameters_num *= kth_order-1;
    }
    
    template < typename T, size_t kth_order >
    pairwise_interaction_tensor_factorization<T, kth_order>::pairwise_interaction_tensor_factorization(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num)
    :base_class(std::move(tensor_A), a_fill_array<size_t, kth_order>(parameters_rank), mini_batch_num)
    ,lambda(lambda)
    {
        //  set weights
        this->weights = weights;
        //  update parameters_num;
        parameters_num *= kth_order-1;
    }
    
    template < typename T, size_t kth_order >
    void pairwise_interaction_tensor_factorization<T, kth_order>::clear()
    {
        base_class::clear();
        weights = a_fill_array<T, kth_order*(kth_order-1)/2>(1);
    }
    
    template < typename T, size_t kth_order >
    void pairwise_interaction_tensor_factorization<T, kth_order>::save_parameters(std::ostream& myout) const
    {
        base_class::save_parameters(myout);
        std::copy(weights.begin(), weights.end(), std::ostream_iterator<T>(myout, " "));
        myout<<std::endl;
    }
    template < typename T, size_t kth_order >
    void pairwise_interaction_tensor_factorization<T, kth_order>::load_parameters(std::istream& myin)
    {
        base_class::load_parameters(myin);
        std::copy_n(std::istream_iterator<T>(myin), kth_order*(kth_order-1)/2, weights.begin());
    }
    template < typename T, size_t kth_order >
    void pairwise_interaction_tensor_factorization<T, kth_order>::test()
    {
        std::cout<<"This is class canonical_decomposition."<<std::endl;
    }
}

#endif
