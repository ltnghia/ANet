#ifndef CAFFE_CAMOURFLAGE_DATA3_LAYER_HPP_
#define CAFFE_CAMOURFLAGE_DATA3_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class CamourflageData3Layer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CamourflageData3Layer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~CamourflageData3Layer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CamourflageData3"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, std::string> > lines_;
  vector<std::string> sal_line_;
  vector<int> label_;
  int lines_id_;
  vector<int> idx_;
};

}  // namespace caffe

#endif  // CAFFE_CAMOURFLAGE_DATA3_LAYER_HPP_
