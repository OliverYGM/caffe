#include "caffe/layers/convert_to_stream.hpp"
#include <vector>
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvertToStreamLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type()
                              << " Layer does not "
                                 "allow in-place computation.";
  N = bottom[0]->num();
  T = bottom[0]->channels();
}

template <typename Dtype>
void ConvertToStreamLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (top.size() != 1) {
    LOG(INFO) << "top size isn't 1 in ConvertToStreamLayer";
    return;
  }
  top[0]->Reshape(T, N, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ConvertToStreamLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int top_offset;
  Dtype* ptop = top[0]->mutable_cpu_data();

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int w = 0; w < bottom[0]->width(); ++w) {
          top_offset = top[0]->offset(c, n, h, w);
          ptop[top_offset] = bottom[0]->data_at(n, c, h, w);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvertToStreamLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_CLASS(ConvertToStreamLayer);
REGISTER_LAYER_CLASS(ConvertToStream);
}  // namespace caffe
