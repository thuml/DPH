#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
__global__ void FocalParameter(const int nthreads, const Dtype threshold, const Dtype* similarity, const Dtype* dot_product, Dtype* focal) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(dot_product[index] > threshold){
        if (similarity[index] == 0){
            focal[index] = 1;
        }
        else{
            focal[index] = 0;
        }
    }
    else if (dot_product[index] < -threshold){
        if (similarity[index] == 0){
            focal[index] = 0;
        }
        else{
            focal[index] = 1;
        }
    }
    else{
        /*
        if (similarity[index] == 0){
            focal[index] = pow(1. / (1+exp(-dot_product[index])), 0.1);
        }
        else{
            focal[index] = pow(1. - (1. / (1+exp(-dot_product[index]))), 0.1);
        }

        Dtype beta = Dtype(1);
        if (similarity[index] == 0){
            focal[index] = pow(beta / (beta + dot_product[index], 0.1);
        }
        else{
            focal[index] = pow(dot_product[index] / (beta + dot_product[index], 0.1)
        }*/

        if (similarity[index] == 0){
            focal[index] = pow((1. + dot_product[index]) / 2., 0.1);
        }
        else{
            focal[index] = pow((1. - dot_product[index]) / 2., 0.1);
        }

       
    }
  }
}

template <typename Dtype>
__global__ void EuclideanDistance(const int nthreads, const int outer_num, const int inner_num, const Dtype* code1, const Dtype* code2, Dtype* distance){
  CUDA_KERNEL_LOOP(index, nthreads) {
    int index_id1 = index / outer_num;
    int index_id2 = index % outer_num;
    distance[index] = 0;
    for (int i = 0; i < inner_num; i++){
      distance[index] += (code1[index_id1 * outer_num + i] - code2[index_id2 * outer_num + i]) * (code1[index_id1 * outer_num + i] - code2[index_id2 * outer_num + i]);
    }
  }
}

template <typename Dtype>
__global__ void CosineDistance(const int nthreads, const int outer_num, const int inner_num, const Dtype* code1, const Dtype* code2, Dtype* distance){
  CUDA_KERNEL_LOOP(index, nthreads) {
    int index_id1 = index / outer_num;
    int index_id2 = index % outer_num;
    distance[index] = 0;
    Dtype length1 = 0;
    Dtype length2 = 0;
    for (int i = 0; i < inner_num; i++){
      length1 += code1[index_id1 * inner_num + i] * code1[index_id1 * inner_num + i];
      length2 += code2[index_id2 * inner_num + i] * code2[index_id2 * inner_num + i];
    }
    length1 = sqrt(length1);
    length2 = sqrt(length2);
    for (int i = 0; i < inner_num; i++){
      distance[index] += (code1[index_id1 * inner_num + i] * code2[index_id2 * inner_num + i]);
    }
    distance[index] = distance[index] / (length1 * length2);
    if(distance[index] >= 1){
      distance[index] = 0.99;
    }
    else if(distance[index] <= -1){
      distance[index] = -0.99;
    }
  }
}

template <typename Dtype>
__global__ void DegreeRowColumn(const int outer_num, const Dtype* similarity, Dtype* row, Dtype* column, Dtype* neg_row, Dtype* neg_column){
  CUDA_KERNEL_LOOP(index, outer_num) { 
    row[index] = 0;
    column[index] = 0;
    for (int i = 0; i < outer_num; i++){
        row[index] += similarity[index*outer_num+i];
        column[index] += similarity[i*outer_num + index];
        neg_row[index] += 1-similarity[index*outer_num+i];
        neg_column[index] += 1-similarity[i*outer_num + index];
    }
    if (row[index] == 0){
        row[index] = 1.;
    }
    else{
        row[index] = Dtype(outer_num) / row[index];
    }
    if (column[index] == 0){
        column[index] = 1.;
    }
    else {
        column[index] = Dtype(outer_num) / column[index];
    }
    if (neg_row[index] == 0){
        neg_row[index] = 1.;
    }
    else {
        neg_row[index] = Dtype(outer_num) / neg_row[index];
    }
    if (neg_column[index] == 0){
        neg_column[index] = 1.;
    }
    else {
        neg_column[index] = Dtype(outer_num) / neg_column[index];
    }
  }
}

template <typename Dtype>
__global__ void SimilarityProcess(const int nthreads, Dtype* similarity, Dtype label_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if((similarity[index] < 0) || (similarity[index] >= label_dim)){
      //unknown label
      similarity[index] = Dtype(-1.0);
    }
    else if(similarity[index] > 0){
      //similar label
      similarity[index] = Dtype(1.0);
    }
  }
}

template <typename Dtype>
__global__ void ContinousSimilarityProcess(const int nthreads, const Dtype* similarity, const Dtype* similarity1, Dtype* similarity2, Dtype* sim, const int outer_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int data_id1 = index / outer_num;
    int data_id2 = index % outer_num;
    sim[index] = similarity[index] * similarity[index] / (similarity1[outer_num*data_id1+data_id1] * similarity2[outer_num*data_id2+data_id2]);
    if(sim[index] == 0){
      sim[index] = 0.25;
    }
  }
}

template <typename Dtype>
__global__ void RemoveZero(const int nthreads, Dtype* similarity1, Dtype* similarity2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity1[index] == 0){
      similarity1[index] = 1.0;
    }
    if(similarity2[index] == 0){
      similarity2[index] = 1.0;
    }
  }
}


template <typename Dtype>
__global__ void PairwiseLossForwardGPU(const int nthreads, const int num, const Dtype* similarity, 
       const Dtype* exp_product, const Dtype* product, const Dtype threshold, Dtype* count, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity[index] >= 0){
      count[index] = Dtype(1.0);
      if((threshold >= 0) && (product[index] >= threshold)){
        loss_data[index] = product[index] * (1 - (similarity[index] > 0));
      }
      else{
        loss_data[index] = log(1 + exp_product[index]) - (similarity[index] > 0) * product[index];
      }
      if(similarity[index] > 0){
        loss_data[index] = loss_data[index];
      }
    }
    else{
      count[index] = Dtype(0.0);
      loss_data[index] = Dtype(0.0);
    }
  }
}


template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* similarity = loss_.mutable_gpu_data();
  Dtype* dot_product = product_.mutable_gpu_data();
  Dtype* exp_product = product_.mutable_gpu_diff();
  Dtype* loss_data = loss_.mutable_gpu_diff();
  Dtype* count = count_.mutable_gpu_data();
  Dtype* similarity1 = own_similarity_.mutable_gpu_data();
  Dtype* similarity2 = own_similarity_.mutable_gpu_diff();
  Dtype* distance = distance_.mutable_gpu_data();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[2]->gpu_data();
  Dtype* label = bottom[1]->mutable_gpu_data();
  Dtype* label1 = bottom[3]->mutable_gpu_data();

  Dtype* row_weight = weight_vector_.mutable_gpu_data();
  Dtype* column_weight = weight_vector_.mutable_gpu_diff();
  Dtype* neg_row_weight = neg_weight_vector_.mutable_gpu_data();
  Dtype* neg_column_weight = neg_weight_vector_.mutable_gpu_diff();

  int nthreads = outer_num_ * outer_num_;

  Dtype loss, count_num;
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label, label1, Dtype(0.0), similarity);
  /*if (continous_similarity_){
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label, label, Dtype(0.0), similarity1);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label1, label1, Dtype(0.0), similarity2);

    RemoveZero<Dtype><<<CAFFE_GET_BLOCKS(own_similarity_.count()), 
      CAFFE_CUDA_NUM_THREADS>>>(own_similarity_.count(), similarity1, similarity2);

    ContinousSimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, similarity1, similarity2, loss_data, outer_num_);
    caffe_gpu_memcpy(nthreads*sizeof(Dtype), loss_data, similarity1);
  }*/

  SimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, label_dim_);

  DegreeRowColumn<Dtype><<<CAFFE_GET_BLOCKS(outer_num_), 
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, similarity, row_weight, column_weight, neg_row_weight, neg_column_weight);

  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, inner_num_, 
      Dtype(1.0), bottom_data, bottom_data1, Dtype(0.0), dot_product);
  caffe_gpu_scal(outer_num_ * outer_num_, sigmoid_param_, dot_product);

  //calculate priority parameter start
  Dtype* focal_parameter = focal_.mutable_gpu_data();

  CosineDistance<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, inner_num_, bottom_data, bottom_data1, distance);

  FocalParameter<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, 20., similarity, distance, focal_parameter);

  Dtype all_focal;
  caffe_gpu_asum(nthreads, focal_parameter, &all_focal);
  ave_focal_ = all_focal / Dtype(nthreads);
  //calculate priority parameter end

  caffe_gpu_exp(outer_num_ * outer_num_, dot_product, exp_product);
  
  PairwiseLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity, exp_product, 
      dot_product, l_threshold_, count, loss_data); 

  //add priority start
  caffe_gpu_mul(nthreads, loss_data, focal_parameter, loss_data);
  caffe_gpu_scal(nthreads, Dtype(1. / ave_focal_), loss_data);
  //add priority end

  caffe_gpu_asum(nthreads, loss_data, &loss);
  caffe_gpu_asum(nthreads, count, &count_num);
  loss /= (count_num > 0 ? count_num : Dtype(1));
  LOG(INFO) << "L loss:" << loss;
  loss = loss * (l_lambda_ > 0);
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
__global__ void PairwiseLossBackwardGPU(const int nthreads, const int num, 
          const Dtype* similarity, const Dtype* exp_product, Dtype* count, Dtype* diff, const Dtype* row_weight, const Dtype* column_weight, const Dtype* neg_row_weight, const Dtype* neg_column_weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int i = index / num;
      int j = index % num;
      if(similarity[index] >= 0){
          diff[index] = 2 * (
              1 / (1 + 1 / exp_product[index]) - 
              (similarity[index] > 0)
          );
          count[index] = Dtype(1.0);
          if(similarity[index] > 0){
              diff[index] = diff[index] * sqrt(row_weight[i] * column_weight[j]);
              count[index] *= sqrt(row_weight[i] * column_weight[j]);
          }
          //else{
          //   diff[index] = diff[index] * sqrt(neg_row_weight[i] * neg_column_weight[j]);
          //    count[index] *= sqrt(neg_row_weight[i] * neg_column_weight[j]);   
          //}
      }
      else{
          diff[index] = Dtype(0.0);
          count[index] = Dtype(0.0);
      }
  }
}


template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* diff = count_.mutable_gpu_diff();
    Dtype* count = count_.mutable_gpu_data();
    const Dtype* similarity = loss_.gpu_data();
    const Dtype* exp_product = product_.gpu_diff();

    const Dtype* similarity1 = own_similarity_.gpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff1 = bottom[2]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_data1 = bottom[2]->gpu_data();

    const Dtype* row_weight = weight_vector_.gpu_data();
    const Dtype* column_weight = weight_vector_.gpu_diff();
    const Dtype* neg_row_weight = neg_weight_vector_.gpu_data();
    const Dtype* neg_column_weight = neg_weight_vector_.gpu_diff();

    int nthreads = outer_num_ * outer_num_;
    
    //calculate diff
    PairwiseLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity,
        exp_product, count, diff, row_weight, column_weight, neg_row_weight, neg_column_weight);
 
    // focal loss
    const Dtype* focal_parameter = focal_.gpu_data();
    caffe_gpu_mul(nthreads, diff, focal_parameter, diff);
    caffe_gpu_scal(nthreads, Dtype(1. / ave_focal_), diff);
    //
       
    /*if(continous_similarity_){
      caffe_gpu_mul(nthreads, diff, similarity1, diff);
      caffe_gpu_scal(nthreads, Dtype(4), diff);
    }*/
    //copy to bottom_diff
    Dtype count_num;
    caffe_gpu_asum(nthreads, count, &count_num);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
        l_lambda_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data1, 
        Dtype(0.0), bottom_diff); 
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
        l_lambda_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data, 
        Dtype(0.0), bottom_diff1);
    caffe_gpu_scal(outer_num_, sigmoid_param_, bottom_diff);
    caffe_gpu_scal(outer_num_, sigmoid_param_, bottom_diff1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLossLayer);

}  // namespace caffe
