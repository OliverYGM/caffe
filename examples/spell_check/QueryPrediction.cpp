#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <caffe/net.hpp>
#include <caffe/util/benchmark.hpp>
#include <caffe/util/io.hpp>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace caffe;

class QueryPrediction {
 public:
  QueryPrediction();
  virtual ~QueryPrediction();
  int init(const std::string& module_path);
  int predict_score(int* input_querys, int batch, int time_steps,
                    float* probvalue);

 private:
  void softMax(caffe::Blob<float>* prob_blob, int batch, int time_steps,
               int* input_querys, const std::vector<int>& query_length,
               float*& probvalue);

  caffe::Net<float>* _net;
};

QueryPrediction::QueryPrediction() : _net(NULL) {}

QueryPrediction::~QueryPrediction() {
  if (NULL != _net) {
    delete _net;
    _net = NULL;
  }
}

int QueryPrediction::init(const std::string& module_path) {
  try {
    Caffe::set_mode(Caffe::CPU);
    string model_path = module_path + "/caffe_module.net_parameter";
    caffe::NetParameter net_param;
    ReadProtoFromBinaryFile(model_path.c_str(), &net_param);
    _net = new caffe::Net<float>(module_path + "/caffe_module.prototxt",
                                 caffe::TEST);
    _net->CopyTrainedLayersFrom(net_param);
  } catch (exception& e) {
    cout << "failed to load caffe net param with exception:[" << e.what() << "]"
         << endl;
    return -1;
  }
  return 0;
}

int QueryPrediction::predict_score(int* input_querys, int batch, int time_steps,
                                   float* probvalue) {
  try {
    vector<int> query_length;  // use to compute probability after forward
    Blob<float>* tmp_input_blob = _net->input_blobs()[0];
    Blob<float>* tmp_cont_blob = _net->input_blobs()[1];
    tmp_input_blob->Reshape(batch, time_steps, 1, 1);
    tmp_cont_blob->Reshape(batch, time_steps, 1, 1);
    float* top_data = tmp_input_blob->mutable_cpu_data();
    float* cont_data = tmp_cont_blob->mutable_cpu_data();
    for (int i = 0; i < batch; i++) {
      int len = 0;
      for (int j = 0; j < time_steps; j++) {
        int idx = time_steps * i + j;
        int val = input_querys[idx];
        top_data[idx] = val;
        if (0 == val) {
          cont_data[idx] = 0;
        } else {
          if (j == 0) {
            cont_data[idx] = 0;
          } else {
            cont_data[idx] = 1;
          }
          len++;
        }
      }
      query_length.push_back(len);
    }
    _net->Forward();
    softMax(_net->output_blobs()[0], batch, time_steps, input_querys,
            query_length, probvalue);
  } catch (exception& e) {
    cout << "predict_score failed with exception:[" << e.what() << "]" << endl;
    return -1;
  }
  return 0;
}

void QueryPrediction::softMax(Blob<float>* prob_blob, int batch, int time_steps,
                              int* input_querys,
                              const vector<int>& query_length,
                              float*& probvalue) {
  const int dim = prob_blob->shape(2);
  float* caffe_output = prob_blob->mutable_cpu_data();
  for (int i = 0; i < batch; i++) {
    int input_querys_idx = i * time_steps;
    float log_sentence_prob = 0.0;
    for (int j = 0; j < time_steps - 1; j++) {
      int next_word_idx = input_querys[++input_querys_idx];
      if (0 == next_word_idx) {
        continue;
      }
      const int base_idx = (j * batch + i) * dim;
      float* caffe_output_t = caffe_output + base_idx;
      float max_num = std::numeric_limits<float>::min();
      for (int k = 0; k < dim; ++k) {
        if (caffe_output_t[k] > max_num) {
          max_num = caffe_output_t[k];
        }
      }
      for (int k = 0; k < dim; ++k) {
        caffe_output_t[k] -= max_num;
      }
      caffe_exp(dim, caffe_output_t, caffe_output_t);
      float denominator_sum = caffe_cpu_asum(dim, caffe_output_t);
      float next_word_prob =
          log(caffe_output_t[next_word_idx] / denominator_sum);
      log_sentence_prob += next_word_prob;
    }
    probvalue[i] = log_sentence_prob / query_length[i];
  }
}

template <typename T>
void printArray(T* values, int size) {
  for (int i = 0; i < size; i++) {
    if (0 != i) {
      cout << ",";
    }
    cout << values[i];
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " module_path test_input" << endl;
    return -1;
  }
  string module_path = argv[1];
  string test_input = argv[2];

  static const int kBatchSize = 3;
  static const int kTimestamp = 20;

  QueryPrediction qp;
  qp.init(module_path);

  vector<vector<int> > lines;
  ifstream in(test_input.c_str());
  string line;
  while (getline(in, line)) {
    vector<string> items;
    boost::split(items, line, boost::is_any_of(","));
    if (kTimestamp != items.size()) {
      cout << "ignore invalid line data:[" << line << "]" << endl;
      continue;
    }
    vector<int> line_data;
    for (int i = 0; i < items.size(); ++i) {
      line_data.push_back(atoi(items[i].c_str()));
    }
    lines.push_back(line_data);
  }
  cout << "test_input:[" << test_input << "] has [" << lines.size()
       << "] valid line" << endl;
  float result[kBatchSize];
  int input_querys[kBatchSize * kTimestamp];
  int query_num = 0;
  Timer timer;
  timer.Start();
  for (int i = 0; i + kBatchSize <= lines.size(); i += kBatchSize) {
    for (int j = 0; j < kBatchSize; j++) {
      memcpy(input_querys + j * kTimestamp, lines[i + j].data(),
             kTimestamp * sizeof(int));
    }
    qp.predict_score(input_querys, kBatchSize, kTimestamp, result);
    query_num++;
  }
  timer.Stop();
  cout << "last query result:[";
  printArray(result, kBatchSize);
  cout << "]" << endl;
  cout << "total time cost:[" << timer.MilliSeconds() << "], query num:["
       << query_num << "], time cost per query:["
       << timer.MilliSeconds() / query_num << "]" << endl;
  return 0;
}

