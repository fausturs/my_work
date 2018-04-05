#include <iostream>
#include <fstream>
#include <utility>

#include "MF.hpp"
#include "TF.hpp"
#include "MySQL.hpp"

void print_tensor(TF::sparse_tensor_tp & A, size_t dim_1, size_t dim_2, size_t dim_3)
{
    for (int i=0; i<dim_1; i++)
    {
        bool flag1 = A.count(i);
        for (int j=0; j<dim_2; j++)
        {
            bool flag2 = flag1 && A[i].count(j);
            for (int k=0; k<dim_3; k++)
            {
                bool flag3 = flag2 && A[i][j].count(k);
                if (flag3)
                    std::cout<< A[i][j][k] <<" ";
                else
                    std::cout<< 0 <<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}


int main(int args, const char* argv[])
{
	/*
    std::ifstream myin("a.txt");
    size_t dim_1, dim_2, dim_3;
    int temp;
    myin>>dim_1>>dim_2>>dim_3;
    TF::sparse_tensor_tp A;
    for (int i=0; i<dim_1; i++)
        for (int j=0; j<dim_2; j++)
            for (int k=0; k<dim_3; k++)
            {
                myin>>temp;
                if (temp!=TF::empty_mark) A[i][j][k]=temp;
            }
    //
    TF tf;
    tf.initialize(2, 2, 2);
    tf.train(A, std::cout);
	
	std::string
	std::ifstream myin("../data/position_cut.txt");
	std::ofstream myout("../data/temp.txt", std::ios::app);
	std::getline(myin, st);
	while (std::getline(myin, st) )
	{
		myout<<st<<std::endl;
	}
	
	//test()
	wjy::MySQL sql("127.0.0.1", "wjy", "", "lagou_data");
	sql.connect();
	sql.query("select id,compname,position from clean_positions");
	auto table = sql.get_result_table();
	for(auto & row : table)
	{
		for (auto s: row) std::cout<<s<<"?";
		std::cout<<"\n";
	}
	std::string st;
	std::ifstream myin("../data/jdid_company_position.txt");
//	std::ofstream myout("../data/temp.txt", std::ios::app);
	while (std::getline(myin, st) )
	{
		auto a = split(st, {'?'});
		if (a.size() !=3) std::cout<<st<<std::endl;
	}
	*/
	auto tensor = create_tensor();
	wjy::save_sparse_tensor(tensor, "../data/tensor.txt");
	std::cout<<"hello world!"<<std::endl;
    return 0;
}


























