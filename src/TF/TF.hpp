#ifndef TF_HPP
#define TF_HPP

#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>
#include <array>
#include <list>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <functional>

#include "func.hpp"
#include "Timer.hpp"
#include "sparse_tensor.hpp"

template<size_t dim>
class TF{
public:
    using element_tp                = double;
    // assume all index start from 0
    using sparse_tensor_index_tp    = wjy::sparse_tensor_index<dim>;
    using sparse_tensor_tp          = wjy::sparse_tensor<double, dim>;
    // default is 0
    static const element_tp empty_mark;
    
    static std::ostringstream TF_log;
private:
    // hyperparameters' rank
    std::array< size_t, dim > parameters_ranks;
    std::array< std::vector<element_tp>, dim > parameters;
    // size of tensor A
    std::array< size_t, dim > tensor_A_dims;
    // tensor s
    std::vector< element_tp > tensor_s;
    //
    element_tp lambda;
    element_tp initial_learning_rate, epsilon;
    size_t max_iter_num, mini_batch_size;
    //
    mutable std::mt19937 mt;
    element_tp rand_range;
    // store iterators for mini-batch sgd
    std::vector< typename sparse_tensor_tp::const_iterator > iterators;
    
	size_t aaaaa;
public:
    TF() = default;
    // movable
    TF(TF&&) = default;
    TF& operator=(TF&&) = default;
    // uncopiable
    TF(const TF&) = delete;
    TF& operator=(const TF&) = delete;
    //
    ~TF() = default;

    // initialize matrix v,u by random real number from [-rand_range, rand_range]
    // if rand_seed = -1 use std::random_device as the seed, else use rand_seed
    void initialize(const std::array< size_t, dim > parameters_ranks, size_t max_iter_num=200000, size_t mini_batch_size=1000, int rand_seed=1, element_tp rand_range=0.2, element_tp lambda=0.5, element_tp initial_learning_rate=0.000001, element_tp epsilon=0.1);
    //
    void train(const sparse_tensor_tp& A, std::ostream& mylog = TF_log);
    //
    element_tp predict(const sparse_tensor_index_tp&) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    void clear();
    // for test
    void print(const sparse_tensor_tp& A);
    
private:
    template<class Input_it1, class Input_it2, class Input_it3, class Output_it>
    void tensor_multiply_vector(Input_it1 tensor_first, size_t d, Input_it2 dims_first, size_t k, Input_it3 vector_first, Output_it out_first) const;
    
    std::vector< element_tp> tensor_s_multiply_n_vector(const sparse_tensor_index_tp&, const std::unordered_set<size_t>& except = {})const;
    //
    void generate_parameters(const sparse_tensor_tp& A);

    element_tp calculate_loss(const sparse_tensor_tp& A) const;
    //
    std::vector< element_tp > calculate_gradient(const sparse_tensor_tp& A) const;
    std::vector< element_tp > calculate_random_gradient(const sparse_tensor_tp& A) const;
    std::vector< element_tp > calculate_s_gradient_at(const sparse_tensor_index_tp&) const;
    
    void update_parameters(const std::vector< element_tp > gradient, element_tp learning_rate);
    //
};

template<size_t dim>
const typename TF<dim>::element_tp TF<dim>::empty_mark = 0;

template<size_t dim>
std::ostringstream TF<dim>::TF_log;

template<size_t dim>
void TF<dim>::initialize(const std::array< size_t, dim > parameters_ranks, size_t max_iter_num, size_t mini_batch_size, int rand_seed, element_tp rand_range, element_tp lambda, element_tp initial_learning_rate, element_tp epsilon)
{
    this->parameters_ranks      = parameters_ranks;
    this->rand_range            = rand_range;
    this->lambda                = lambda;
    this->initial_learning_rate = initial_learning_rate;
    this->max_iter_num          = max_iter_num;
    this->mini_batch_size       = mini_batch_size;
    this->epsilon               = epsilon;
    
    if (rand_seed == -1)
        mt = std::mt19937( std::random_device{}() );
    else
        mt = std::mt19937( rand_seed );
    
    clear();
}

template<size_t dim>
void TF<dim>::train(const sparse_tensor_tp& A, std::ostream& mylog)
{
    if (A.size() == 0) return;
    // generate parameter u,v,w and s
    generate_parameters(A);
    //
    mylog<<"Start training..."<<std::endl;
    format_print(mylog, "epoch", "loss", "gradient_norm", "time(s)");
    format_print(mylog, 0, (long long)calculate_loss(A), "unknow", 0);
	wjy::Timer timer;
	timer.start();
    size_t epoch_size = 5000;
    for (size_t iter = 0; iter<max_iter_num; iter++)
    {
        // gradient descent
        //auto gradient = calculate_gradient(A);
		aaaaa = iter;

        auto gradient = calculate_random_gradient(A);
		size_t temp = iter/(50*epoch_size);
		element_tp half = std::pow(2, temp);
		auto learning_rate = initial_learning_rate / (half);
        update_parameters(gradient, learning_rate);
        // print log
        if ( (iter % epoch_size)==(epoch_size-1) )
        {
			//auto gradient = calculate_gradient(A);
			//update_parameters(gradient, learning_rate);
            auto gradient_norm = std::inner_product(gradient.begin(), gradient.end(),gradient.begin(),static_cast<element_tp>(0));
			timer.end();
            format_print(mylog, iter/epoch_size+1, (long long)calculate_loss(A), gradient_norm, timer.get_duration_s());
            // stop condition
            if ( std::sqrt(gradient_norm) < epsilon ) break;
        }
		//std::cout<<" 4 "<<std::endl;
        //if (iter == 1)break;
    }
    mylog<<"Finished!"<<std::endl;
	auto gradient = calculate_gradient(A);
	auto gradient_norm = std::inner_product(gradient.begin(), gradient.end(),gradient.begin(),static_cast<element_tp>(0));
	mylog<<"gradient norm:"<<gradient_norm<<std::endl;
    //
    iterators.clear();
}

template<size_t dim>
typename TF<dim>::element_tp TF<dim>::predict(const sparse_tensor_index_tp& indexes) const
{
    auto ans = tensor_s_multiply_n_vector(indexes, {});
    return ans[0];
}

template<size_t dim>
void TF<dim>::save(const std::string& path) const
{
    std::ofstream myout(path);
    assert(myout);
    
    auto save_vector = [&myout](auto& ve){
        std::copy(ve.begin(), ve.end(), std::ostream_iterator<element_tp>(myout, " "));
    };
    save_vector(tensor_A_dims);
    save_vector(parameters_ranks);
    for (auto & p : parameters) save_vector(p);
    save_vector(tensor_s);
    myout.close();
}

template<size_t dim>
void TF<dim>::load(const std::string& path)
{
    std::ifstream myin(path);
    assert(myin);
    
    clear();
    auto load_vector = [&myin](auto& ve, size_t n){
        ve.reserve(n);
        std::copy_n(std::istream_iterator<element_tp>(myin), n, std::back_inserter(ve));
    };
    load_vector(tensor_A_dims, dim);
    load_vector(parameters_ranks, dim);
    size_t s_length = 1;
    for (size_t i=0; i<dim; i++)
    {
        load_vector(parameters[i], tensor_A_dims[i]*parameters_ranks[i]);
        s_length *= parameters_ranks[i];
    }
    load_vector(tensor_s, s_length);
    myin.close();
}

template<size_t dim>
void TF<dim>::clear()
{
    for (auto & parameter : parameters) parameter.clear();
    tensor_s.clear();
    iterators.clear();
}

// for test
template<size_t dim>
void TF<dim>::print(const sparse_tensor_tp& A)
{
	element_tp max = 0;
	int i = 0;
    for (auto & a : A) 
	{
		auto pre = std::abs(predict(a.first));
		//	if (pre > 10) std::cout<<++i<<std::endl;
		max = std::max(max, pre);
	}
	std::cout<<max<<std::endl;
	auto l = calculate_loss(A);
	std::cout<<(long long)l<<std::endl;
}

template<size_t dim>
template<class Input_it1, class Input_it2, class Input_it3, class Output_it>
void TF<dim>::tensor_multiply_vector(Input_it1 tensor_first, size_t d, Input_it2 dims_first, size_t k, Input_it3 vector_first, Output_it out_first) const
{
    // B_{pre, suf} = \sum_{i_k} A_{pre, i_k, suf} * v_{i_k};
    size_t pre_size = 1, suf_size = 1;
    for (size_t i=0; i<k ;i++,dims_first++) pre_size*=(*dims_first);
    size_t dim_k = *(dims_first++);
    for (size_t i=k+1; i<d ;i++,dims_first++) suf_size *= (*dims_first);
    std::vector< element_tp > temp(dim_k);
    for (size_t pre=0; pre<pre_size; pre++)
    {
        for (size_t suf=0; suf<suf_size; suf++)
        {
            size_t index = pre*(suf_size*dim_k) + suf;
            for (size_t i=0; i<dim_k; i++,index+=suf_size) temp[i] = *(tensor_first+index);
            *(out_first++) = std::inner_product(temp.begin(), temp.end(), vector_first, static_cast<element_tp>(0) );
        }
    }
}

template<size_t dim>
std::vector< typename TF<dim>::element_tp > TF<dim>::tensor_s_multiply_n_vector(const sparse_tensor_index_tp& indexes, const std::unordered_set<size_t>& except) const
{
    std::vector< size_t > dims(parameters_ranks.begin(), parameters_ranks.end());
    auto dims_rit = dims.rbegin();
    std::vector< element_tp > t1( tensor_s );
    std::vector< element_tp > t2( t1.size() );
    for (size_t i=0; i<dim; i++,dims_rit++)
    {
        if (except.count(dim-i-1)) continue;
        auto v_it = parameters[dim-i-1].begin() + parameters_ranks[dim-i-1]*indexes[dim-i-1];
        tensor_multiply_vector(t1.begin(), dim, dims.begin(), dim-i-1, v_it, t2.begin());
        std::swap(t1, t2);
        *dims_rit = 1;
    }
    size_t length = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});
    t1.resize(length);
    return t1;
}

//
template<size_t dim>
void TF<dim>::generate_parameters(const sparse_tensor_tp& A)
{
    tensor_A_dims = {0};
    for (auto & a : A)
    {
        auto & indexes = a.first;
        for (size_t i=0; i<dim; i++) tensor_A_dims[i] = std::max(tensor_A_dims[i], indexes[i]+1);
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
    size_t s_length = 1;
    for (size_t i=0; i<dim; i++)
    {
        rand_generate_vector(parameters[i], parameters_ranks[i]*tensor_A_dims[i]);
        s_length *= parameters_ranks[i];
    }
    rand_generate_vector(tensor_s, s_length);
    //
    size_t n = A.size() / mini_batch_size;
    iterators.resize(n+1);
    auto it = A.begin();
    for (size_t i=0; i<n; i++)
    {
        iterators[i] = it;
        for (size_t j=0; j<mini_batch_size; j++) it++;
    }
    iterators[n] = A.end();
}

template<size_t dim>
typename TF<dim>::element_tp TF<dim>::calculate_loss(const sparse_tensor_tp& A) const
{
    element_tp loss = 0;
    for (auto & parameter : parameters)
        loss = std::inner_product(parameter.begin(), parameter.end(), parameter.begin(), loss);
    loss = std::inner_product(tensor_s.begin(), tensor_s.end(), tensor_s.begin(), loss);
    loss *= lambda;
    for (auto & a : A)
    {
        auto & indexes = a.first;
        auto & value = a.second;
        auto pred = predict( indexes );
        loss += (pred - value)*(pred - value);
    }
    return loss;
}

// if denote A.size() as n, denote tensor_s.size() as m
// this function's complexity is O(n*m*dim)
template<size_t dim>
std::vector< typename TF<dim>::element_tp > TF<dim>::calculate_gradient(const sparse_tensor_tp& A) const
{
    std::vector< element_tp > gradient;
    size_t temp = tensor_s.size();
    for (auto & parameter : parameters) temp+= parameter.size();
    gradient.resize(temp, 0);
    // F-norm's gradient
    auto gradient_it1 = gradient.begin(), gradient_it2 = gradient_it1;
    for (auto & parameter : parameters)
    {
        gradient_it2 = gradient_it1 + parameter.size();
        add_to(gradient_it1, gradient_it2, parameter.begin(), 2*lambda);
        gradient_it1 = gradient_it2;
    }
    gradient_it2 = gradient_it1 + tensor_s.size();
    add_to(gradient_it1, gradient_it2, tensor_s.begin(), 2*lambda);
    
    // gradient of \sum(A - s*para)^2
    for (auto & a : A)
    {
        auto & indexes = a.first;
        auto & value = a.second;
        element_tp pred = predict( indexes );
        auto temp = (pred - value)*2;
        gradient_it1 = gradient.begin();
        for (size_t i=0; i<dim; i++)
        {
            auto g_v = tensor_s_multiply_n_vector(indexes, {i});
            gradient_it2 = gradient_it1 + parameters_ranks[i]*indexes[i];
            add_to(gradient_it2, gradient_it2+parameters_ranks[i], g_v.begin(), temp);
            gradient_it1 += parameters[i].size();
        }
        auto g_s = calculate_s_gradient_at(indexes);
        add_to(gradient_it1, gradient.end(), g_s.begin(), temp);
    }
    return gradient;
}

template<size_t dim>
std::vector< typename TF<dim>::element_tp > TF<dim>::calculate_random_gradient(const sparse_tensor_tp& A) const
{
    // initialize gradient;
	//if (aaaaa == 3070) std::clog<<"random gradient 1 ";
    std::vector< element_tp > gradient;
    size_t temp = tensor_s.size();
    for (auto & parameter : parameters) temp+= parameter.size();
    gradient.resize(temp, 0);
    // tensor_s's F-norm's gradient
	//if (aaaaa == 3070) std::clog<<" 2 ";
    add_to(gradient.rbegin(), gradient.rbegin()+tensor_s.size(), tensor_s.rbegin(), 2*lambda);
    // random a mini-batch
	//if (aaaaa == 3070) std::clog<<" 3 ";
    std::uniform_int_distribution<size_t> uid(0, iterators.size()-2);
    auto sample = uid(mt);
    std::unordered_set< size_t > flag;
	//if (aaaaa == 3070) std::clog<<" sample: "<<sample<<" size: "<<iterators.size();
    for (auto it = iterators[sample]; it!= iterators[sample+1]; it++)
    {
        auto & indexes = it->first;
        auto & value = it->second;
        element_tp pred = predict( indexes );
        auto temp = (pred - value)*2;
        auto gradient_it1 = gradient.begin(), gradient_it2 = gradient_it1;
        for (size_t i=0; i<dim; i++)
        {
            auto g_v = tensor_s_multiply_n_vector(indexes, {i});
            gradient_it2 = gradient_it1 + parameters_ranks[i]*indexes[i];
            add_to(gradient_it2, gradient_it2+parameters_ranks[i], g_v.begin(), temp);
            auto dis = std::distance(gradient.begin(), gradient_it2);
            // v's F-norm's gradient
            if (flag.count(dis)==0)
            {
                add_to(gradient_it2, gradient_it2+parameters_ranks[i],
                       parameters[i].begin() + parameters_ranks[i]*indexes[i],
                       2*lambda
                       );
                flag.insert(dis);
            }
            gradient_it1 += parameters[i].size();
        }
        auto g_s = calculate_s_gradient_at(indexes);
        add_to(gradient_it1, gradient.end(), g_s.begin(), temp);
    }
	//if (aaaaa == 3070) std::clog<<" end "<<std::endl;
    return gradient;
}

// if denote tensor_s.size() as m
// the function is O(m*dim)
template<size_t dim>
std::vector< typename TF<dim>::element_tp > TF<dim>::calculate_s_gradient_at(const sparse_tensor_index_tp& indexes) const
{
    std::vector< element_tp > gradient_s( tensor_s.size(), 1);
    size_t pre_size = 1, dim_i = 1;
    size_t suf_size = tensor_s.size();
    for (size_t i=0; i<dim; i++)
    {
        dim_i = parameters_ranks[i];
        suf_size /= dim_i;
        pre_size = tensor_s.size() / (suf_size * dim_i);
        auto v_begin = parameters[i].begin() + parameters_ranks[i]*indexes[i];
        for (size_t pre=0; pre<pre_size; pre++)
            for (size_t k=0; k<dim_i; k++)
                for (size_t suf=0; suf<suf_size; suf++)
                {
                    gradient_s[pre*(suf_size*dim_i)+k*(suf_size)+suf] *= (*(v_begin+k));
                }
    }
    return gradient_s;
}

template<size_t dim>
void TF<dim>::update_parameters(const std::vector< element_tp > gradient, element_tp learning_rate)
{
    auto gradient_it = gradient.begin();
	element_tp ma=-100, mi=100;
    for (size_t i=0; i<dim; i++)
    {
        auto & parameter = parameters[i];
        add_to(parameter.begin(), parameter.end(), gradient_it, -learning_rate);
		auto m1 = std::max_element(parameter.begin(), parameter.end());
		auto m2 = std::min_element(parameter.begin(), parameter.end());
		ma = std::max(ma, *m1); mi = std::min(mi, *m2);
        gradient_it += parameter.size();
    }
    add_to(tensor_s.begin(), tensor_s.end(), gradient_it, -learning_rate);
	auto m1 = std::max_element(tensor_s.begin(), tensor_s.end());
	auto m2 = std::min_element(tensor_s.begin(), tensor_s.end());
	ma = std::max(ma, *m1); mi = std::min(mi, *m2);
	//std::cout<<"x "<<ma<<" "<<mi<<std::endl;
}

#endif
