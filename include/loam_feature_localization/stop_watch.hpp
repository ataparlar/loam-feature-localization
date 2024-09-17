#ifndef BUILD_SRC_LOAM_FEATURE_LOCALIZATION_INCLUDE_LOAM_FEATURE_LOCALIZATION_STOP_WATCH_HPP_
#define BUILD_SRC_LOAM_FEATURE_LOCALIZATION_INCLUDE_LOAM_FEATURE_LOCALIZATION_STOP_WATCH_HPP_

#include <chrono>
#include <ratio>
#include <iostream>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::high_resolution_clock::time_point;

class StopWatch {
private:
  using clock = std::chrono::high_resolution_clock;
  using time_point = std::chrono::time_point<clock>;

  time_point time_point_start;
  time_point time_point_end;
  bool is_running;
  bool is_reset;

public:
  StopWatch() : is_running(false), is_reset(true) {}

  void reset() {
    time_point_start = time_point_end;
    is_reset = true;
    is_running = false;
  }

  void start() {
    if (is_reset) {
      time_point_start = clock::now();
      is_running = true;
    }
    is_reset = false;
    is_running = true;
  }

  void stop() {
    time_point_end = clock::now();
    is_running = false;
  }

  double elapsed_milliseconds() {
    if (is_running)
      time_point_end = clock::now();
    std::chrono::duration<double, std::milli> elapsed_duration =
      time_point_end - time_point_start;
    return elapsed_duration.count();
  }

  double elapsed_microseconds() {
    if (is_running)
      time_point_end = clock::now();
    std::chrono::duration<double, std::micro> elapsed_duration =
      time_point_end - time_point_start;
    return elapsed_duration.count();
  }

  void print_ms(const std::string& str_before) {
    std::cout << str_before << elapsed_milliseconds() << " ms." << std::endl;
  }
};


#endif  // BUILD_SRC_LOAM_FEATURE_LOCALIZATION_INCLUDE_LOAM_FEATURE_LOCALIZATION_STOP_WATCH_HPP_
