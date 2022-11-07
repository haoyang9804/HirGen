#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern "C" int run(float *input, float *output, int input_size, int output_size) {
  Eigen::ThreadPool tp(std::thread::hardware_concurrency());
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
  Graph graph;
  graph.set_thread_pool(&device);

  std::copy(input, input + input_size, graph.arg0_data());
  auto ok = graph.Run();
  if (not ok) return -1;
  std::copy(graph.result0_data(), graph.result0_data() + output_size, output);
  return 0;
}
