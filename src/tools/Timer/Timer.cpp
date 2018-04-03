//
//  Timer.cpp
//  test_c++
//
//  Created by wjy on 16/7/10.
//  Copyright © 2016年 wjy. All rights reserved.
//

#include "Timer.hpp"

using namespace wjy;


void Timer::start()
{
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::end()
{
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
}

long long Timer::get_duration_ns()
{
    return duration.count();
}

long long Timer::get_duration_us()
{
    return (std::chrono::duration_cast<std::chrono::microseconds>(duration)).count();
}

long long Timer::get_duration_ms()
{
    return (std::chrono::duration_cast<std::chrono::milliseconds>(duration)).count();
}
long long Timer::get_duration_s()
{
    return (std::chrono::duration_cast<std::chrono::seconds>(duration)).count();
}
long long Timer::get_duration_minute()
{
    return (std::chrono::duration_cast<std::chrono::minutes>(duration)).count();
}
long long Timer::get_duration_hour()
{
    return (std::chrono::duration_cast<std::chrono::hours>(duration)).count();
}

