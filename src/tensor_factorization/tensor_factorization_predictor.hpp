#ifndef tensor_factorization_predictor_hpp
#define tensor_factorization_predictor_hpp

#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <cassert>

#include "sparse_tensor.hpp"
#include "predictor.hpp"

namespace wjy {
    template< typename T, size_t kth_order >
    class tensor_factorization_predictor : public predictor<T, sparse_tensor_index<kth_order> >{
        using base_class = predictor<T, sparse_tensor_index<kth_order> >;
    protected:
        //  the tensor needed factorization
        sparse_tensor<T, kth_order> tensor_A;
        //  all the TF<T, k> model have k matirx parameters.
        std::array< std::pair<size_t, size_t> ,kth_order> matrixes_size;
        //  some value for sgd
        std::vector< typename sparse_tensor<T, kth_order>::const_iterator > batchs;
        std::unordered_map< size_t, size_t > counter;
    public:
        tensor_factorization_predictor() = default;
        tensor_factorization_predictor(sparse_tensor<T, kth_order>&& tensor_A, const std::array<size_t, kth_order>& parameters_rank, size_t mini_batch_num);
        //  movable
        tensor_factorization_predictor(tensor_factorization_predictor&&) = default;
        tensor_factorization_predictor& operator=(tensor_factorization_predictor&&) = default;
        //  uncopiable, write nothing, compiler implicitly assume it's delete
        
        virtual void clear()override;
        virtual void save_parameters(std::ostream&) const override;
        virtual void load_parameters(std::istream&) override;
        virtual void test() override;
        
        virtual ~tensor_factorization_predictor() override = default;
    };
    
    template< typename T, size_t kth_order>
    tensor_factorization_predictor<T, kth_order>::tensor_factorization_predictor(sparse_tensor<T, kth_order>&& tensor_A, const std::array<size_t, kth_order>& parameters_rank, size_t mini_batch_num)
    :base_class(mini_batch_num)
    {
        //  init tensor_A
        this->tensor_A = std::move(tensor_A);
        //  init matrixes_size
        for (size_t i=0; i<kth_order; i++) matrixes_size[i] = {0, parameters_rank[i]};
        for (auto & enter : this->tensor_A)
        {
            auto & index = enter.first;
            for (size_t i=0; i<kth_order; i++)
                matrixes_size[i].first = std::max(matrixes_size[i].first, index[i]+1);
        }
        //  update parameters_num
        for (auto & size : matrixes_size) this->parameters_num += size.first * size.second;
        //  init batchs and counter, need matrixes_size is ready.
        //  if mini_batch_num == 0 we assume this predict don't need a stochastic trainer.
        if (mini_batch_num == 0) return;
        size_t mini_batch_size = this->tensor_A.size() / mini_batch_num;
        size_t temp = this->tensor_A.size() % mini_batch_num;
        batchs.resize(mini_batch_num + 1);
        auto it = this->tensor_A.cbegin();
        for (size_t i=0; i<mini_batch_num; i++)
        {
            batchs[i] = it;
            size_t this_size = mini_batch_size;
            if (temp!=0) { this_size++; temp--; }
            for (size_t j=0; j<this_size; j++)
            {
                auto & index = it->first;
                size_t offset = 0;
                for (size_t k=0; k<kth_order; k++)
                {
                    counter[offset + index[k]*matrixes_size[k].second]++;
                    offset += matrixes_size[k].first*matrixes_size[k].second;
                }
                it++;
            }
        }
        batchs[mini_batch_num] = it;
    }
    
    template< typename T, size_t kth_order>
    void tensor_factorization_predictor<T, kth_order>::clear()
    {
        tensor_A.clear();
        for (auto & p : matrixes_size) p = {0, 0};
        batchs.clear();
        counter.clear();
        base_class::clear();
    }
    
    template< typename T, size_t kth_order>
    void tensor_factorization_predictor<T, kth_order>::save_parameters(std::ostream& myout) const
    {
        base_class::save_parameters(myout);
        for (auto & p : matrixes_size) myout<<p.first<<" "<<p.second<<" ";
        myout<<std::endl;
    }
    
    template< typename T, size_t kth_order>
    void tensor_factorization_predictor<T, kth_order>::load_parameters(std::istream& myin)
    {
        base_class::load_parameters(myin);
        for (auto & p : matrixes_size) myin>>p.first>>p.second;
    }
    
    template< typename T, size_t kth_order>
    void tensor_factorization_predictor<T, kth_order>::test()
    {
         std::cout<<"This is class tensor_factorization_predictor."<<std::endl;
    }

}

#endif /* end of file tensor_factorization_predictor.hpp */
