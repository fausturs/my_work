//
//  MySQL.cpp
//  mysql_lab
//
//  Created by wjy on 16/4/10.
//  Copyright © 2016年 wjy. All rights reserved.
//

#include "MySQL.hpp"
//#include <iostream>

using namespace wjy;

MySQL::MySQL(std::string ip,std::string usr_name,std::string pwd,std::string database_name)
{
    this->ip = ip;
    this->usr_name = usr_name;
    this->pwd = pwd;
    this->database_name = database_name;
    //
    is_connected = false;
    //
    table_head.clear();
    table.clear();
    error_info.clear();
    //初始化数据库
    mysql_init(&mysql);
}

MySQL::~MySQL()
{
    disconnect();
}


//连接数据库
int MySQL::connect()
{
    error_info.clear();
    MYSQL *connection;
    connection = mysql_real_connect(&mysql,
                                    ip.c_str(),
                                    usr_name.c_str(),
                                    pwd.c_str(),
                                    database_name.c_str(),
                                    0,0,0);
    if (connection == NULL)
    {
        error_info =  mysql_error(&mysql);
        return 1;
    }
    //数据库连接成功了
    is_connected = true;
	//设置返回字符串的编码
	query("set names utf8;");
    return 0;
}
//断开
int MySQL::disconnect()
{
    error_info.clear();
    if (is_connected == false)
    {
        error_info =  "It was not connect with mysql!";
        return 1;
    }
    //该函数没有返回值  所以我们认为断开成功了
    mysql_close(&mysql);
    is_connected = false;
    table.clear();
    table_head.clear();
    return 0;
}

//
bool MySQL::is_connect()
{
    return is_connected;
}


//
int MySQL::query(std::string st)
{
    error_info.clear();
    MYSQL_RES *result;
    MYSQL_ROW sql_row;
    MYSQL_FIELD *fd;
    if (is_connected == false)
    {
        error_info =  "It was not connect with mysql!";
        return 1;
    }
    int res = mysql_query(&mysql, st.c_str());
    if (res == 0)
    {
        result = mysql_store_result(&mysql);//保存查询到的数据到result
        //清除上次查询的结果  换句话说 只要执行成功 mysql不管有没有返回值
        //上次查询的结果都会被抛弃
        table.clear();
        table_head.clear();
        if (result == NULL)
        {
            return 0;
        }
        unsigned long  rows = mysql_num_rows(result);
        unsigned long  cols = mysql_num_fields(result);
        //获取列名
        for (unsigned long i = 0;i<cols;i++)
        {
            fd=mysql_fetch_field(result);
            table_head.push_back(fd->name);
//            std::cout<<fd->name<<std::endl;
        }
        //获取整张表
        table.resize(rows);
        for (unsigned long i = 0;i<rows;i++)
        {
            sql_row = mysql_fetch_row(result);
            for (unsigned long j = 0;j<cols;j++)
            {
                table[i].push_back(sql_row[j]);
            }
        }
        mysql_free_result(result);
    }
    else
    {
        error_info =  mysql_error(&mysql);
        return 1;
    }
    return 0;
}



//这两个函数返回mysql查询的结果  一个返回表头  一个返回表内容
const std::vector<std::string>& MySQL::get_result_table_head() const
{
    return table_head;
}
const std::vector< std::vector<std::string> >& MySQL::get_result_table() const
{
    return table;
}


const std::string& MySQL::get_error_info() const
{
    return error_info;
}


//结果表的行和列的数量
unsigned long MySQL::get_table_rows()
{
    return table.size();
}
unsigned long MySQL::get_table_cols()
{
    if (table.size()>0)
    {
        return table[0].size();
    }
    return 0;
}

//返回的bool值表示是否已经连接到一个数据库
bool MySQL::operator ()()
{
    return is_connected;
}
//
MySQL::operator bool()
{
    return is_connected;
}
//
const std::vector< std::string >& MySQL::operator [](unsigned long i) const
{
    return table[i];
}









