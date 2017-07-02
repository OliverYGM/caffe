#ifndef CAFFE_CONVERT_TO_STREAM_LAYER_HPP_
#define CAFFE_CONVERT_TO_STREAM_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ConvertToStreamLayer : public Layer<Dtype> {
 public:
  explicit ConvertToStreamLayer(const LayerParameter& param)
      : Layer<Dtype>(param), T(0), N(0) {}
  virtual ~ConvertToStreamLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConvertToStream"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  int T;
  int N;
};

}  // namespace caffe
#endif  // CAFFE_CONVERT_TO_STREAM_LAYER_HPP_
