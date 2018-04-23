#ifndef my_model_20180422_hpp
#define my_model_20180422_hpp

#include "tensor_factorization_predictor.hpp"

template<typename T, size_t kth_order>
class my_model_20180422 final: public tensor_factorization_predictor<T, kth_order>{

public:
	my_model_20180422() = default;

	virtual ~my_model_20180422() override = default;
	
};

#endif