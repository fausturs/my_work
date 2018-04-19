#include "func.hpp"

#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>
#include <string>
#include <list>
#include <cassert>

#include "sparse_tensor.hpp"


std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

std::unordered_map< std::string, size_t >   demand_level;
std::unordered_map< int, std::pair<std::string, std::string> >  jdid_company_position;

std::unordered_map< std::string, int > skill_counter, company_counter, position_counter;
//

std::pair< std::string, std::string > get_company_position(int id);
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level);

void read_raw_data();
void prepare_filter();
bool skill_fliter(const std::string& skill);
bool company_fliter(const std::string& company);
bool position_fliter(const std::string& position);
void save_new_data(const wjy::sparse_tensor<double, 3> t);

void create_tensor()
{
    read_raw_data();
    prepare_filter();
    
    std::ifstream myin("../data/raw_data/position_cut.txt");
    assert(myin);
    wjy::sparse_tensor<double, 3> tensor;
    wjy::sparse_tensor_index<3> index = {0, 0, 0};
    std::string st_jd;
    while (std::getline(myin, st_jd))
    {
        auto job_description = wjy::split(st_jd, {',','/'});
        int id = std::atoi( job_description[0].c_str() );
        std::pair<std::string, std::string> company_position;
		if (jdid_company_position.find(id) == jdid_company_position.end()) continue;
		company_position = jdid_company_position[id];
        // these two keys must in the map.
        index[0] = company_to_id[company_position.first];
        index[1] = position_to_id[company_position.second];
        if ((company_fliter(company_position.first) && position_fliter(company_position.second))==false)  continue;
        auto demand_level_list = parse_jd_demand(job_description, skill_to_id, demand_level);
        for (auto& skill_level : demand_level_list)
        {
            index[2] = skill_to_id[skill_level.first];
            if (!skill_fliter(skill_level.first)) continue;
            tensor[ index ] = skill_level.second;
        }
		//if (id > 30) break;
    }
	myin.close();
    //
    save_new_data(tensor);
}




/****************************************************************************************/
/*                      private functions&var for this file                             */
/****************************************************************************************/

//wjy::MySQL sql("127.0.0.1", "wjy", "", "lagou_data");


// it's a (something, something_id) pair list. something_id helps to build the tensor.
void read_some_list(const std::string& path, std::unordered_map< std::string, size_t >& some_to_id, std::vector< std::string >& id_to_some)
{
    std::ifstream myin(path);
    assert(myin);
    some_to_id.clear();
    id_to_some.clear();
    
    std::string something;
    size_t n = 0;
    while (std::getline(myin, something))
    {
		if (some_to_id.find(something) != some_to_id.end()) continue;
        some_to_id[something] = n++;
        id_to_some.push_back(something);
    }
	myin.close();
}

void read_skill_list(const std::string& path)
{
    return read_some_list(path, skill_to_id, id_to_skill);
}

void read_company_list(const std::string& path)
{
    return read_some_list(path, company_to_id, id_to_company);
}

void read_position_list(const std::string& path)
{
    return read_some_list(path, position_to_id, id_to_position);
}

void read_demand_level(const std::string& path = "../data/demand_level.txt")
{
    std::ifstream myin(path);
    assert(myin);
    demand_level.clear();
    std::string word;
    int level;
    while (myin>>word>>level) demand_level[word] = level;
	myin.close();
}

void read_jdid_company_position(const std::string& path = "../data/raw_data/jdid_company_position.txt")
{
    std::ifstream myin(path);
    assert(myin);
    jdid_company_position.clear();
    std::string st;
    while(std::getline(myin, st))
	{
		auto sts = wjy::split(st, {'?'});
		int id = std::atoi(sts[0].c_str());
		assert(id>0 && sts.size()==3);
		jdid_company_position[id] = std::make_pair(sts[1], sts[2]);
	}
	myin.close();
}

void read_raw_data()
{
    read_skill_list("../data/raw_data/skill_list.txt");
    read_company_list("../data/raw_data/company_list.txt");
    read_position_list("../data/raw_data/position_list.txt");
    read_demand_level();
    read_jdid_company_position();
}

void read_all(wjy::Date data_date)
{
    auto date = data_date.to_string("");
    read_skill_list("../data/skill_list_"+date+".txt");
    read_company_list("../data/company_list_"+date+".txt");
    read_position_list("../data/position_list_"+date+".txt");
//    read_demand_level();
//    read_jdid_company_position();
}


// input a jd, output all skills and it's demand.
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level)
{
    std::list< std::pair<std::string, int> > skill_level_list;
	std::unordered_set< std::string > skills;
    auto i = jd.size(), j = i;
    do{
        i--;
        auto skill = skill_list.find(jd[i]);
        if (skill == skill_list.end() || skills.find(skill->first)!=skills.end()) continue;
        if (j>i) // two skills may have the same level of demand
            for (j=i-1; j>0; j--)
			{
                if (demand_level.find(jd[j]) != demand_level.end()) break;
			}
		if (j==0) break;
        skill_level_list.push_back( std::make_pair(jd[i], demand_level.at(jd[j])) );
		skills.insert(skill->first);
    }while (i!=1);// jd[0] is the id for this jd
    return skill_level_list;
}

void prepare_filter()
{
    skill_counter.clear();
    company_counter.clear();
    position_counter.clear();
    for (auto &cp : jdid_company_position)
    {
        auto & com = cp.second.first;
        auto & pos = cp.second.second;
        company_counter[com]++;
        position_counter[pos]++;
    }
    std::ifstream myin("../data/raw_data/position_cut.txt");
    assert(myin);
    std::string st_jd;
    while (std::getline(myin, st_jd))
    {
        auto job_description = wjy::split(st_jd, {',','/'});
        for (auto & word : job_description)
            if (skill_to_id.find(word)!=skill_to_id.end()) skill_counter[word]++;
    }
    myin.close();
}
bool skill_fliter(const std::string& skill)
{
    return (skill_counter[skill] >= 10);
}
bool company_fliter(const std::string& company)
{
    return (company_counter[company] >= 20);
}
bool position_fliter(const std::string& position)
{
    return (position_counter[position] >= 10);
}
void save_new_data(const wjy::sparse_tensor<double, 3> t)
{
    std::unordered_map< std::string, size_t > skill_to_new_id, company_to_new_id, position_to_new_id;
    auto save_something = [](auto fliter, auto & old_map, auto & new_map, const auto & path){
        std::ofstream myout(path);
        assert(myout);
        size_t n = 0;
        for (auto & p : old_map)
        {
            if (!fliter(p.first)) continue;
            new_map[p.first] = n++;
            myout<<p.first<<std::endl;
        }
        myout.close();
    };
    auto today = wjy::Date{}.to_string("");
    auto path = "../data/"+today+"/";
    save_something(skill_fliter, skill_to_id, skill_to_new_id, path+"skill_list_"+today+".txt");
    save_something(company_fliter, company_to_id, company_to_new_id, path+"company_list_"+today+".txt");
    save_something(position_fliter, position_to_id, position_to_new_id, path+"position_list_"+today+".txt");
    
    wjy::sparse_tensor<double, 3> new_tensor;
    wjy::sparse_tensor_index<3> new_index;
    for (auto & p : t)
    {
        auto & old_index = p.first;
        new_index[0] = company_to_new_id[ id_to_company[ old_index[0] ] ];
        new_index[1] = position_to_new_id[ id_to_position[ old_index[1] ] ];
        new_index[2] = skill_to_new_id[ id_to_skill[ old_index[2] ] ];
        new_tensor[new_index] = p.second;
    }
    wjy::save_sparse_tensor(new_tensor, path+"tensor_dim3_"+today+".txt");
}


/*
std::pair< std::string, std::string > get_company_position(int id)
{
	std::cout<<id<<std::endl;
    if (!sql) sql.connect();
    std::string q = "select compname,position from clean_positions where id = \'"+ std::to_string(id) + "\';";
    sql.query(q);
    auto rslt = sql.get_result_table();
    return std::make_pair( rslt[0][0], rslt[0][1] );
}
*/


//for test
void test()
{
	//read_all();
	//create_tensor();
}




