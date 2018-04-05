#ifndef TF1_HPP
#define TF1_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>

#include "func.hpp"


class TF1{
public:
    using element_tp            = double;
    // assume all index start from 0
    using sparse_tensor_tp      = std::unordered_map< size_t, std::unordered_map< size_t, std::unordered_map< size_t, element_tp> > >;
    // default is 0
    static const element_tp empty_mark;
    
    static std::ostringstream TF1_log;
private:
    // hyperparameter
    size_t u_rank, v_rank, w_rank;
    // size of tensor A
    size_t dim_1, dim_2, dim_3;
    //
    element_tp lambda;
    element_tp initial_learning_rate, epsilon;
    size_t max_iter_num;
    //
    std::mt19937 mt;
    element_tp rand_range;
    // model parameters
    // here u,v,w is matrix, s is tensor. so s \in R^{u_rank \times v_rank \times w_rank}
    // we store all this as a vector
    // the index translate rule for u,v,w is (i,j)->(i*?_rank+j)
    // for tensor s is (i,j,k)->(i*(v_rank+w_rank) + j*(w_rank) + k)
    std::vector< element_tp > u,v,w;
    std::vector< element_tp > s;
    // when we need calculate tensor s multiply with some vector.
    // the way of we store s is important. especially when training the model.
    // here is two different way, but the same tensor with s.
    // the only place we need these is calculate gradient, it's useless when predict
    mutable std::vector< element_tp > s2, s3;

public:
    TF1() = default;
    // movable
    TF1(TF1&&) = default;
    TF1& operator=(TF1&&) = default;
    // uncopiable
    TF1(const TF1&) = delete;
    TF1& operator=(const TF1&) = delete;
    //
    ~TF1() = default;

    // initialize matrix v,u by random real number from [-rand_range, rand_range]
    // if rand_seed = -1 use std::random_device as the seed, else use rand_seed
    void initialize(size_t u_rank=3, size_t v_rank=3, size_t w_rank=3, int rand_seed=1, element_tp rand_range=2.0, element_tp lambda=0.5, element_tp initial_learning_rate=0.01, size_t max_iter_num=20000, element_tp epsilon=0.1);
    //
    template<class Tensor_tp>
    void train(const Tensor_tp& A, std::ostream& mylog = TF1_log);
    //
    element_tp predict(size_t i, size_t j, size_t k) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    void clear();
    // for test
    void print();
    
private:
    //
    void generate_u_v_w_s(const sparse_tensor_tp& A);
    void update_s2_s3() const;
    //
    // element_tp get_Aijk(const sparse_tensor_tp& A, size_t i, size_t j, size_t k) const;
    //
    element_tp calculate_loss(const sparse_tensor_tp& A) const;
    //
    std::vector< element_tp > calculate_gradient(const sparse_tensor_tp& A) const;
    //
    template<class T, class Input_it1, class Input_it2>
    std::vector< element_tp > tensor_mult_two_vec(T s, Input_it1 first1, size_t n1, Input_it2 first2, size_t n2) const;
    
    template<class Input_it1, class Input_it2, class Input_it3>
    std::vector< element_tp > calculate_s_gradient(Input_it1 first1, Input_it2 first2, Input_it3 first3) const;
};

/*
 template member function
 */
template <class Tensor_tp>
void TF1::train(const Tensor_tp& A, std::ostream& mylog)
{
    if (A.size() == 0) return;
    // generate parameter u,v,w and s
    generate_u_v_w_s(A);
    //
    mylog<<"Start training..."<<std::endl;
    format_print(mylog, "epoch", "loss", "gradient_norm", "time");
    format_print(mylog, 0, calculate_loss(A), "unknow", 0);
    size_t epoch_size = 100;
    for (size_t iter = 0; iter<max_iter_num; iter++)
    {
        // gradient descent
        auto gradient = calculate_gradient(A);
        auto learning_rate = initial_learning_rate / (iter+1);
        auto gradient_it = gradient.begin();
        add_to(u.begin(), u.end(), gradient_it              , -learning_rate);
        add_to(v.begin(), v.end(), gradient_it+=dim_1*u_rank, -learning_rate);
        add_to(w.begin(), w.end(), gradient_it+=dim_2*v_rank, -learning_rate);
        add_to(s.begin(), s.end(), gradient_it+=dim_3*w_rank, -learning_rate);
        // print log
        if ( (iter % epoch_size)==(epoch_size-1) )
        {
            auto gradient_norm = std::inner_product(gradient.begin(), gradient.end(),gradient.begin(),static_cast<element_tp>(0));
            format_print(mylog, iter/epoch_size+1, calculate_loss(A), gradient_norm, 0);
            // stop condition
            if ( std::sqrt(gradient_norm) < epsilon ) break;
        }
//        if (iter == 5)break;
    }
    mylog<<"Finished!"<<std::endl;
}

template<class T, class Input_it1, class Input_it2>
std::vector< TF1::element_tp > TF1::tensor_mult_two_vec(T s, Input_it1 first1, size_t n1, Input_it2 first2, size_t n2) const
{
    auto n = s.size();
    element_tp zero = 0;
    std::vector< element_tp > temp1(n/n1), temp2(n/(n1*n2));
    using vector_it = std::vector< element_tp >::iterator;
    //
    inner_product_n(s.begin(), s.end(), first1, n1, temp1.begin(), zero);
    inner_product_n(temp1.begin(), temp1.end(), first2, n2, temp2.begin(), zero);
    
    return temp2;
}

template<class Input_it1, class Input_it2, class Input_it3>
std::vector< TF1::element_tp > TF1::calculate_s_gradient(Input_it1 first1, Input_it2 first2, Input_it3 first3) const
{
    auto n1 = u_rank, n2 = v_rank, n3 = w_rank;
    std::vector< element_tp > temp(n1*n2*n3);
    for (int i=0; i<n1; i++)
        for (int j=0; j<n2; j++)
            for (int k=0; k<n3; k++)
                temp[i*(n2*n3) + j*(n3) + k] = (*(first1+i))*(*(first2+j))*(*(first3+k));
    return temp;
}


#endif
