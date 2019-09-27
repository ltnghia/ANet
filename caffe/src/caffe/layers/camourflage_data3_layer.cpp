#ifdef USE_OPENCV
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/camourflage_data3_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CamourflageData3Layer<Dtype>::~CamourflageData3Layer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void CamourflageData3Layer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "CamourflageData3Layer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  int idx=0;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
	string salfn = "";
	int labelfn = -1;
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
	  iss >> labelfn;
	  iss >> salfn;
    }
    lines_.push_back(std::make_pair(imgfn, segfn));
	sal_line_.push_back(salfn);
	label_.push_back(labelfn);

    idx_.push_back(idx);
    idx++;

LOG(INFO) << imgfn;
LOG(INFO) << segfn;
LOG(INFO) << salfn;
LOG(INFO) << labelfn;

  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  switch (ExactNumTopBlobs()){
	  case 1:
		this->output_labels_ = false;
		this->output_mask_ = false;
		this->output_truth_ = false;
		break;
	  case 2:
		this->output_labels_ = true;
		this->output_mask_ = false;
		this->output_truth_ = false;
		break;
	  case 3:
		this->output_labels_ = true;
		this->output_mask_ = false;
		this->output_truth_ = true;
		break;
	  case 4:
		this->output_labels_ = true;
		this->output_mask_ = true;
		this->output_truth_ = true;
		break;
  }

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  //const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_width = 0;
  int crop_height = 0;
  CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
	|| (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
    << "Must either specify crop_size or both crop_height and crop_width.";
  if (transform_param.has_crop_size()) {
    crop_width = transform_param.crop_size();
    crop_height = transform_param.crop_size();
  } 
  if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
    crop_width = transform_param.crop_width();
    crop_height = transform_param.crop_height();
  }

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_width > 0 && crop_height > 0) {
    top[0]->Reshape(batch_size, channels, crop_height, crop_width);
    this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(batch_size, channels, crop_height, crop_width);
    }

	if(this->output_labels_){
		top[1]->Reshape(batch_size, 1, crop_height, crop_width);
		this->transformed_label_.Reshape(batch_size, 1, crop_height, crop_width);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
		  this->prefetch_[i]->label_.Reshape(batch_size, 1, crop_height, crop_width);
		}
	}
		
	if(this->output_mask_){
		top[3]->Reshape(batch_size, 1, crop_height, crop_width);
		this->transformed_mask_.Reshape(batch_size, 1, crop_height, crop_width);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
		  this->prefetch_[i]->mask_.Reshape(batch_size, 1, crop_height, crop_width);
		}
	}

  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(batch_size, channels, height, width);
    }

    if(this->output_labels_){
		top[1]->Reshape(batch_size, 1, height, width);
		this->transformed_label_.Reshape(batch_size, 1, height, width);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
		  this->prefetch_[i]->label_.Reshape(batch_size, 1, height, width);
		}
	}
	
	if(this->output_mask_){
		top[3]->Reshape(batch_size, 1, height, width);
		this->transformed_mask_.Reshape(batch_size, 1, height, width);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
		  this->prefetch_[i]->mask_.Reshape(batch_size, 1, height, width);
		}
	}
  }
  
	//label
	if(this->output_truth_){
		top[2]->Reshape(batch_size, 1, 1, 1);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->truth_.Reshape(batch_size, 1, 1, 1);
		}
	}
  

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
		
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
		
  if(this->output_truth_){
	  LOG(INFO) << "output classification size: " << top[2]->num() << ","
	    << top[2]->channels() << "," << top[2]->height() << ","
	    << top[2]->width();
  }
  
  if(this->output_mask_){
	  LOG(INFO) << "output mask size: " << top[3]->num() << ","
	    << top[3]->channels() << "," << top[3]->height() << ","
	    << top[3]->width();
  }
}

template <typename Dtype>
void CamourflageData3Layer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(idx_.begin(), idx_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void CamourflageData3Layer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data     = batch->data_.mutable_cpu_data();
  Dtype* top_label    = batch->label_.mutable_cpu_data(); 
  Dtype* top_truth;
  Dtype* top_mask;
  
  if(this->output_truth_){
	  top_truth = batch->truth_.mutable_cpu_data(); 
  }
  
  if(this->output_mask_){
	  top_mask = batch->mask_.mutable_cpu_data(); 
  }

  const int max_height = batch->data_.height();
  const int max_width  = batch->data_.width();

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  const int lines_size = lines_.size();
  int lines_id;

    //LOG(INFO) << "lines_size: " << lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    lines_id = idx_[lines_id_];

    std::vector<cv::Mat> cv_img_seg;
    cv::Mat img, seg, mask;

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id);

	//LOG(INFO) << root_folder + lines_[lines_id].first;

    img = ReadImageToCVMat(root_folder + lines_[lines_id].first,
	  new_height, new_width, is_color);
    cv_img_seg.push_back(img);

    // TODO(jay): implement resize in ReadImageToCVMat

    if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id].first;
    }
    if (label_type == ImageDataParameter_LabelType_PIXEL) {

	//LOG(INFO) << root_folder + lines_[lines_id].second;

      seg = ReadLabelToCVMat(root_folder + lines_[lines_id].second,
					    new_height, new_width);
      cv_img_seg.push_back(seg);
      if (!cv_img_seg[1].data) {
	DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id].second;
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id].second.c_str());
      //LOG(INFO) << "label: " << label;
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);      
    }
    else {
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, 
		  CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }

	if(this->output_truth_){
		top_truth[item_id] = label_[lines_id];
	}
	
	if(this->output_mask_){
		cv_img_seg.push_back(ReadImageToCVMat(root_folder + sal_line_[lines_id],
					    new_height, new_width, false));
	}

//LOG(INFO) << label_[lines_id];
//LOG(INFO) << root_folder + sal_line_[lines_id];


    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;
    offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);
	
	if(this->output_mask_){
		offset = batch->mask_.offset(item_id);
		this->transformed_mask_.set_cpu_data(top_mask + offset);
		
		this->data_transformer_->TransformImgAndSeg(cv_img_seg, 
		&(this->transformed_data_), &(this->transformed_label_),
		&(this->transformed_mask_), ignore_label);
	} else {
		this->data_transformer_->TransformImgAndSeg(cv_img_seg, 
		&(this->transformed_data_), &(this->transformed_label_),
		ignore_label);
	}

        //LOG(INFO) << "image size: " << img.rows << " - " << img.cols;
	//LOG(INFO) << "label size: " << seg.rows << " - " << seg.cols;

    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(CamourflageData3Layer);
REGISTER_LAYER_CLASS(CamourflageData3);

}  // namespace caffe
#endif  // USE_OPENCV
