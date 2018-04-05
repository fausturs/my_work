#include "TF.hpp"

#include <fstream>
#include <iterator>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cassert>

/*
 some local function
 */

/*
 member value or member function of class MF
 */
const TF::element_tp TF::empty_mark = 0;
std::ostringstream TF::TF_log;

void TF::initialize(size_t u_rank, size_t v_rank, size_t w_rank, int rand_seed, element_tp rand_range, element_tp lambda, element_tp initial_learning_rate, size_t max_iter_num, element_tp epsilon)
{
    this->u_rank = u_rank;
    this->v_rank = v_rank;
    this->w_rank = w_rank;
    this->rand_range = rand_range;
    this->lambda = lambda;
    this->initial_learning_rate = initial_learning_rate;
    this->max_iter_num = max_iter_num;
    this->epsilon = epsilon;
    
    if (rand_seed == -1)
        mt = std::mt19937( std::random_device{}() );
    else
        mt = std::mt19937( rand_seed );
    
    clear();
}

TF::element_tp TF::predict(size_t i, size_t j, size_t k) const
{
    // here use s_w_v_u denote s*w*v*u, it's a scale. in fact u,v,w denote u_i,v_j,w_k
    // similar s_w_v means s*w*v, it's a vector
    element_tp s_w_v_u = 0;
    auto s_w_v = tensor_mult_two_vec(s, w.begin()+w_rank*k, w_rank, v.begin()+v_rank*j, v_rank);
    s_w_v_u = std::inner_product(s_w_v.begin(), s_w_v.end(), u.begin()+u_rank*i, s_w_v_u);
    return s_w_v_u;
}

void TF::save(const std::string& path) const
{
    std::ofstream myout(path);
    assert(myout);
    
    myout<<u_rank<<" "<<v_rank<<" "<<w_rank<<" ";
    myout<<dim_1<<" "<<dim_2<<" "<<dim_3<<" ";
    auto save_vector = [&myout](auto& ve){
        std::copy(ve.begin(), ve.end(), std::ostream_iterator<element_tp>(myout, " "));
    };
    save_vector(u);
    save_vector(v);
    save_vector(w);
    save_vector(s);
}

void TF::load(const std::string& path)
{
    std::ifstream myin(path);
    assert(myin);
    
    clear();
    myin>>u_rank>>v_rank>>w_rank;
    myin>>dim_1>>dim_2>>dim_3;
    auto load_vector = [&myin](auto& ve, size_t n){
        ve.reserve(n);
        std::copy_n(std::istream_iterator<element_tp>(myin), n, std::back_inserter(ve));
    };
    load_vector(u, dim_1 * u_rank);
    load_vector(v, dim_2 * v_rank);
    load_vector(w, dim_3 * w_rank);
    load_vector(s, u_rank * v_rank * w_rank);
}

void TF::clear()
{
    u.clear();
    v.clear();
    w.clear();
    s.clear();s2.clear();s3.clear();
}

void TF::print()
{
    auto & myout = std::cout;
    auto save_vector = [&myout](auto& ve){
        std::copy(ve.begin(), ve.end(), std::ostream_iterator<element_tp>(myout, " "));
    };
//    save_vector(u);
//    save_vector(v);
//    save_vector(w);
    save_vector(s);
    myout<<std::endl;
}

void TF::generate_u_v_w_s(const sparse_tensor_tp& A)
{
    dim_1 = dim_2 = dim_3 = 0;
    for (auto & A_i : A)
    {
        dim_1 = std::max(dim_1, A_i.first+1);
        for (auto & A_ij : A_i.second)
        {
            dim_2 = std::max(dim_2, A_ij.first+1);
            for (auto & A_ijk : A_ij.second) dim_3 = std::max(dim_3, A_ijk.first+1);
        }
    }
    clear();
    // gen_rand() generate a real number uniformly from [-rand_range, rand_range],
    // and use mt as the random device
    std::uniform_real_distribution<element_tp> urd(-rand_range, rand_range);
    auto gen_rand = std::bind(urd, std::ref(mt));
    auto rand_generate_vector = [&gen_rand](auto& ve, size_t n){
        ve.reserve(n);
        std::generate_n(std::back_inserter(ve), n, gen_rand);
    };
    rand_generate_vector(u, dim_1 * u_rank);
    rand_generate_vector(v, dim_2 * v_rank);
    rand_generate_vector(w, dim_3 * w_rank);
    rand_generate_vector(s, u_rank * v_rank * w_rank);
    
    //generate s2,s3 by s
    s2.resize(s.size());
    s3.resize(s.size());
}

// s2,s3 is mutable variable, so const member function can change it.
void TF::update_s2_s3() const
{
    for (size_t i=0; i<u_rank; i++)
        for (size_t j=0; j<v_rank; j++)
            for (size_t k=0; k<w_rank; k++)
            {
                auto value = s[i*(v_rank*w_rank) + j*(w_rank) + k];
                s2[j*(u_rank*w_rank) + i*(w_rank) + k] = value;
                s3[k*(u_rank*v_rank) + i*(v_rank) + j] = value;
            }
}

TF::element_tp TF::calculate_loss(const TF::sparse_tensor_tp& A) const
{
    element_tp loss = 0;
    loss = std::inner_product(u.begin(), u.end(), u.begin(), loss);
    loss = std::inner_product(v.begin(), v.end(), v.begin(), loss);
    loss = std::inner_product(w.begin(), w.end(), w.begin(), loss);
    loss = std::inner_product(s.begin(), s.end(), s.begin(), loss);
    loss *= lambda;
    
    for (auto & A_i : A)
        for (auto & A_ij : A_i.second)
            for (auto & A_ijk : A_ij.second)
            {
                size_t i=A_i.first, j=A_ij.first, k=A_ijk.first;
                auto pred = predict(i, j, k);
                loss += (pred - A_ijk.second)*(pred - A_ijk.second);
            }
    return loss;
}

std::vector< TF::element_tp > TF::calculate_gradient(const TF::sparse_tensor_tp& A) const
{
    std::vector< element_tp > gradient(dim_1*u_rank + dim_2*v_rank + dim_3*w_rank + u_rank*v_rank*w_rank, 0);
    auto g_u_first = gradient.begin();
    auto g_v_first = g_u_first + dim_1*u_rank;
    auto g_w_first = g_v_first + dim_2*v_rank;
    auto g_s_first = g_w_first + dim_3*w_rank, g_s_last = gradient.end();
    add_to(g_u_first, g_v_first, u.begin(), 2*lambda);
    add_to(g_v_first, g_w_first, v.begin(), 2*lambda);
    add_to(g_w_first, g_s_first, w.begin(), 2*lambda);
    add_to(g_s_first, g_s_last , s.begin(), 2*lambda);
    // s maybe changed in gradient descent, s2 and s3 need update.
    update_s2_s3();
    for (auto & A_i : A)
        for (auto & A_ij : A_i.second)
            for (auto & A_ijk : A_ij.second)
            {
                size_t i=A_i.first, j=A_ij.first, k=A_ijk.first;
                auto g_ui_first = g_u_first + i*u_rank;
                auto g_vj_first = g_v_first + j*v_rank;
                auto g_wk_first = g_w_first + k*w_rank;
                //
                auto u_v_w = calculate_s_gradient(u.begin()+i*u_rank, v.begin()+j*v_rank, w.begin()+k*w_rank);
                auto s_w_v = tensor_mult_two_vec(s , w.begin()+w_rank*k, w_rank, v.begin()+v_rank*j, v_rank);
                auto s_w_u = tensor_mult_two_vec(s2, w.begin()+w_rank*k, w_rank, u.begin()+u_rank*i, u_rank);
                auto s_v_u = tensor_mult_two_vec(s3, v.begin()+v_rank*j, v_rank, u.begin()+u_rank*i, u_rank);
                element_tp temp = 0;
                temp = std::inner_product(s_w_v.begin(), s_w_v.end(), u.begin()+u_rank*i, temp);
                temp = (temp - A_ijk.second)*2;
                // update u,v,w's gradient
                add_to(g_ui_first, g_ui_first+u_rank, s_w_v.begin(), temp);
                add_to(g_vj_first, g_vj_first+v_rank, s_w_u.begin(), temp);
                add_to(g_wk_first, g_wk_first+w_rank, s_v_u.begin(), temp);
                // update s's gradient
                add_to(g_s_first, g_s_last, u_v_w.begin(), temp);
            }
    return gradient;
}


