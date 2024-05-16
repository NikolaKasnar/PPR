#pragma once

#include <chrono>
#include <string>

// Klasa za mjerenje vremena. Ne treba dirati. 
class Clock{
    public:
        enum Interval{sec, millisec, microsec, nanosec};
        Clock(){start();}
        void start(){ begin = mClock.now();}
        std::pair<double, std::string> stop(Interval unit){ 
            using namespace std::chrono;
            end = mClock.now();
            switch(unit){
                case sec:      return {duration_cast<seconds>(end-begin).count(), " sec"};
                case millisec: return {duration_cast<milliseconds>(end-begin).count(), " ms"};
                case microsec: return {duration_cast<microseconds>(end-begin).count(), " us"};
                default:       return {duration_cast<nanoseconds>(end-begin).count(), " ns"};
            };
        }
    private:
        using SystemClock = std::chrono::high_resolution_clock;
        SystemClock mClock;
        std::chrono::time_point<SystemClock> begin;
        std::chrono::time_point<SystemClock> end;
};
