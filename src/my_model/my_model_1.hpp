#ifndef my_model_1_hpp
#define my_model_1_hpp

#include "pairwise_interaction_tensor_factorization.hpp"

#include <algorithm>

namespace wjy{

    //  classify the first order of tensor to several group, assume in each group they are similar.

    template<typename T, size_t kth_order>
    class my_model_1 : public pairwise_interaction_tensor_factorization<T, kth_order>{
    protected:
    	using base_class = pairwise_interaction_tensor_factorization<T, kth_order >;
        using base_class::tensor_A;
        using base_class::matrixes_size;
        using base_class::batchs;
        using base_class::counter;
        using base_class::parameters;
        using base_class::parameters_num;
        using base_class::lambda;
        using base_class::weights;

        virtual std::vector<T> gradient(const std::vector<T>&) const override;
        virtual std::vector<T> mini_batch_stochastic_gradient(const std::vector<T>&, size_t) const override;
        virtual T loss(const std::vector<T>&) const override;

        //  category parameters
        T lambda_1;
        size_t category_num;
        std::vector<size_t> category_map;
        void init_category_map(std::vector<size_t>&&);

    public:
    	my_model_1() = default;
    	//  all vector in this model is the same length, so parameters_rank is a scalar.
        //  weights default = 1
        my_model_1(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1, size_t mini_batch_num);
        //  init weights by a given array
        my_model_1(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1, const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num);
            
    	//	movable
    	my_model_1(my_model_1&&) = default;
    	my_model_1& operator=(my_model_1&&) = default;
    	//  uncopiable, write nothing, compiler implicitly assume it's delete 

        virtual void test() override;
    	virtual ~my_model_1() override = default;
    };

    template<typename T, size_t kth_order>
    std::vector<T> my_model_1<T, kth_order>::gradient(const std::vector<T>& para) const
    {
        auto g = base_class::gradient(para);
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        std::vector<T> temp(vector_length, 0);
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                //
                size_t offset1 = offset_ij + k*vector_length;
                size_t offset2 = category_parameters_offset + category_num*vector_length*(j-1) + vector_length*category_map[k];
                auto it1 = para.begin() + offset1;
                auto it2 = para.begin() + offset2;
                vector_sum(it1, static_cast<T>(2), it2, static_cast<T>(-2), vector_length, temp.begin());
                //  update gradient
                auto g_it1 = g.begin() + offset1;
                auto g_it2 = g.begin() + offset2;
                vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_1);
                vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_1);
            }
        }
        return g;
    }

    template<typename T, size_t kth_order>
    std::vector<T> my_model_1<T, kth_order>::mini_batch_stochastic_gradient(const std::vector<T>& para, size_t batch_i) const 
    {
        //  这里新加入的这部分norm不好做随机，因为他是两个参数的差的norm
        //  所以在这里，直接将这部分norm分为n份，minibatch大小是多大就加上多少份。
        auto g = base_class::mini_batch_stochastic_gradient(para, batch_i);
        size_t n = tensor_A.size();
        //  这一行的时间按复杂度是O(m)
        size_t m = std::distance(batchs[batch_i], batchs[batch_i+1]);
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        std::vector<T> temp(vector_length, 0);
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                //
                size_t offset1 = offset_ij + k*vector_length;
                size_t offset2 = category_parameters_offset + category_num*vector_length*(j-1) + vector_length*category_map[k];
                auto it1 = para.begin() + offset1;
                auto it2 = para.begin() + offset2;
                vector_sum(it1, static_cast<T>(2), it2, static_cast<T>(-2), vector_length, temp.begin());
                //  update gradient
                auto g_it1 = g.begin() + offset1;
                auto g_it2 = g.begin() + offset2;
                //  唯一和上面那个函数gradient不一样的地方在这里后面的系数
                vector_sum(g_it1, g_it1+vector_length, temp.begin(), lambda_1*m/n);
                vector_sum(g_it2, g_it2+vector_length, temp.begin(), -lambda_1*m/n);
            }
        }
        return g;
    }

    template<typename T, size_t kth_order>
    T my_model_1<T, kth_order>::loss(const std::vector<T>& para) const 
    {
        T l = base_class::loss(para), l1 = 0;
        size_t vector_length = matrixes_size.front().second;
        size_t category_parameters_offset = 0;
        for (auto & m : matrixes_size) category_parameters_offset += m.first*m.second*(kth_order-1);
        size_t i = 0;
        for (size_t j=1; j<kth_order; j++)
        {
            size_t offset_ij = matrixes_size[i].first*vector_length*(j-1);
            for (size_t k=0; k<matrixes_size[i].first; k++)
            {
                auto it1 = para.begin() + offset_ij + k*vector_length;
                auto it2 = para.begin() + category_parameters_offset + category_num*vector_length*(j-1) + vector_length*category_map[k];
                l1 = std::inner_product(it1, it1+vector_length, it2, l1);
            }
        }
        return l + l1*lambda_1;
    }

    template<typename T, size_t kth_order>
    void my_model_1<T, kth_order>::init_category_map(std::vector<size_t>&& m)
    {
        category_map = std::move(m);
        //  discretization
        auto v = category_map;
        std::sort(v.begin(), v.end());
        auto it_end = std::unique(v.begin(), v.end());
        category_num = std::distance(v.begin(), it_end);
        for (auto & x : category_map)
        {
            auto it = std::lower_bound(v.begin(), it_end, x);
            x = std::distance(v.begin(), it);
        }
    }


    template<typename T, size_t kth_order>
    my_model_1<T, kth_order>::my_model_1(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1, size_t mini_batch_num)
    :base_class(std::move(tensor_A), parameters_rank, lambda, mini_batch_num)
    ,lambda_1(lambda_1)
    {
        //
        init_category_map(std::move(category_map));
        //  update parameters_num
        parameters_num += parameters_rank*category_num*(kth_order-1);
    }

    template<typename T, size_t kth_order>
    my_model_1<T, kth_order>::my_model_1(sparse_tensor<T, kth_order>&& tensor_A, size_t parameters_rank, T lambda, std::vector<size_t> category_map, T lambda_1, const std::array<T, kth_order*(kth_order-1)/2>& weights ,size_t mini_batch_num)
    :base_class(std::move(tensor_A), parameters_rank, lambda, mini_batch_num)
    ,lambda_1(lambda_1)
    {
        //
        init_category_map(std::move(category_map));
        //  update parameters_num
        parameters_num += parameters_rank*category_num*(kth_order-1);
    }

    template<typename T, size_t kth_order>
    void my_model_1<T, kth_order>::test()
    {
        std::cout<<"This is class my_model_1."<<std::endl;
    }

}


#endif