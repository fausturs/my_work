#ifndef MF_HPP
#define MF_HPP

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>

#include "func.hpp"
/*
 
 */
class MF{
public:
    using element_tp                = double;
    // assume matrix index start from 0
    using dense_matrix_tp           = std::vector< std::vector<element_tp> >;
    using sparse_matrix_tp          = std::unordered_map<size_t, std::unordered_map<size_t, element_tp> >;
    // default is 0
    static const element_tp empty_mark;
    
    static std::ostringstream MF_log;
private:
    // \min \sum_{A_{i,j}\neq 0} (A_{i,j} - u_{i,all} * v_j{all,i})^2 + \lambda(\|u\|_F^2 + \|v\|_F^2)
    // k is rank of matrix u and v
    size_t rows_num, cols_num, k;
    element_tp lambda;
    element_tp initial_learning_rate, epsilon;
    size_t max_iter_num;
    //
    std::mt19937 mt;
    element_tp rand_range;
    // assume A \in R^{n\times m}
    // here we let u \in R^{n \times k} v \in R^{m \times k}
    // so A \approxeq u \times v^\top
    // dense_matrix_tp u, v we store it as vector
    // the index translate rule for u is (i,j)->(k*i+j), for v is (i,j)->(k*j+i)
    std::vector< element_tp > u, v;
public:
    MF() = default;
    // movable
    MF(MF&&) = default;
    MF& operator=(MF&&) = default;
    // uncopiable
    MF(const MF&) = delete;
    MF& operator=(const MF&) = delete;
    
    
    // initialize matrix v,u by random real number from [-rand_range, rand_range]
    // if rand_seed = -1 use std::random_device as the seed, else use rand_seed
    void initialize(size_t k = 3, int rand_seed = 1, element_tp rand_range = 2.0, element_tp lambda = 0.5, element_tp initial_learning_rate = 0.05, size_t max_iter_num = 2000, element_tp epsilon = 0.1);
    // we assume the input matrix A has nozero cols or rows.
    // or this function will not converge
    template<class Matrix_tp>
    void train(const Matrix_tp& A, std::ostream& mylog = MF_log);
    // assume 0<=i<n and 0<=j<m
    // it's undefined when i or j out of range
    element_tp predict(size_t i, size_t j) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    void clear();
    // for test
    void print();
private:
    void generate_u_v(const dense_matrix_tp& A);
    void generate_u_v(const sparse_matrix_tp& A);
    //
    element_tp get_Aij(const dense_matrix_tp& A, size_t i, size_t j) const;
    element_tp get_Aij(const sparse_matrix_tp& A, size_t i, size_t j) const;
    // loss function is:
    // \sum_{i,j, A_{i,j}\neq 0} (A_{i,j} - u_{i,all} * v_j{all,i}) + \lambda(\|u\|_F^2 + \|v\|_F^2)
    template <class Matrix_tp>
    element_tp calculate_loss(const Matrix_tp& A) const;
//    element_tp calculate_loss(const sparse_matrix_tp& A) const;
    //
    template<class Matrix_tp>
    std::vector< element_tp > calculate_gradient(const Matrix_tp& A) const;
//    std::vector< element_tp > calculate_gradient(const sparse_matrix_tp& A) const;
};




/*
 template functions
 */

template<class Matrix_tp>
void MF::train(const Matrix_tp& A, std::ostream& mylog)
{
    if (A.size() == 0) return;
    // generate matrix u and v randomly
    generate_u_v(A);
    //
    auto format_print = [&mylog](auto a, auto b, auto c){
        mylog<<std::left<<std::setw(16)<<a<<std::left<<std::setw(16)<<b;
        mylog<<std::left<<std::setw(16)<<c<<std::endl;
    };
    mylog<<"Start training..."<<std::endl;
    format_print("epoch", "loss", "time");
    format_print(0, calculate_loss(A), 0);
    size_t epoch_size = 100;
    for (size_t iter = 0; iter<max_iter_num; iter++)
    {
        // gradient descent
        auto gradient = calculate_gradient(A);
        auto learning_rate = initial_learning_rate / (iter+1);
        add_to(u.begin(), u.end(), gradient.begin(), -learning_rate);
        add_to(v.begin(), v.end(), gradient.begin()+rows_num*k, -learning_rate);
        // print log
        if ( (iter % epoch_size)==(epoch_size-1) ) format_print(iter/epoch_size+1, calculate_loss(A), 0);
        // stop condition
        auto gradient_norm = std::inner_product(gradient.begin(), gradient.end(),
                                                gradient.begin(),
                                                static_cast<element_tp>(0)
                                                );
        if ( std::sqrt(gradient_norm) < epsilon ) break;
        
    }
    mylog<<"Finished!"<<std::endl;
}

template <class Matrix_tp>
MF::element_tp MF::calculate_loss(const Matrix_tp& A) const
{
    element_tp loss = 0;
    loss = std::inner_product(u.begin(), u.end(), u.begin(), loss);
    loss = std::inner_product(v.begin(), v.end(), v.begin(), loss);
    loss *= lambda;
    // in here, loss = \lambda( \|u\|_F^2 + \|v\|_F^2 )
    element_tp temp;
    for (size_t i=0; i<rows_num; i++)
    {
        for (size_t j=0; j<cols_num; j++)
        {
            auto A_ij = get_Aij(A, i, j);
            if ( A_ij==empty_mark ) continue;
            temp = A_ij - predict(i, j);
            loss += temp * temp;
        }
    }
    return loss;
}

// u is a matrix which has rows_num rows and k cols
// we calculate the i th row's gradient by the follow equation:
// (u_i)' = \lambda * 2 u_i + 2 \sum_{A_{i,j}\neq 0} (u_i * v_j - A_{i,j})v_j
// and for matrix v, is similar.
template <class Matrix_tp>
std::vector< MF::element_tp > MF::calculate_gradient(const Matrix_tp& A) const
{
    std::vector< element_tp > gradient((rows_num+cols_num)*k, 0);
    auto u_g_first = gradient.begin(), u_g_last = u_g_first + rows_num*k;
    auto v_g_first = u_g_last , v_g_last = gradient.end();;
    add_to(u_g_first, u_g_last, u.begin(), lambda*2);
    add_to(v_g_first, v_g_last, v.begin(), lambda*2);
    //
    for (size_t i=0; i<rows_num; i++)
    {
        for (size_t j=0; j<cols_num; j++)
        {
            auto A_ij = get_Aij(A, i , j);
            if (A_ij==empty_mark) continue;
            auto ui_g_first = u_g_first + k*i, vj_g_first = v_g_first + k*j;
            auto ui_first = u.begin() + k*i, vj_first = v.begin() + k*j;
            element_tp temp = 0;
            temp = std::inner_product(ui_first, ui_first+k, vj_first, temp);
            temp = (temp - A_ij) * 2;
            add_to(ui_g_first, ui_g_first+k, vj_first, temp);
            add_to(vj_g_first, vj_g_first+k, ui_first, temp);
        }
    }
    return gradient;
}



#endif
