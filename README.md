This is a repository that contains computer vision algorithms that works in rainy conditions. 

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[[paper]()] [[code]()]

# Surveys

* A Comprehensive Benchmark Analysis of Single Image Deraining: Current Challenges and Future Perspectives (IJCV 2021) [[paper]([https://link.springer.com/article/10.1007/s11263-020-01416-w](https://link.springer.com/content/pdf/10.1007/s11263-020-01416-w.pdf?pdf=button%20sticky))]
* Survey on rain removal from videos or a single image (SCI China 2021) [[paper](https://link.springer.com/content/pdf/10.1007/s11432-020-3225-9.pdf?pdf=button)]
* Single image deraining: From model-based to data-driven and beyond (TPAMI 2020) [[pdf](https://arxiv.org/abs/1912.07150)]
* A Survey of Single Image De-raining in 2020 (Arxiv 2020) [[paper](https://link.springer.com/chapter/10.1007/978-981-16-3945-6_75)]

* Keypoints of these surveys
  * Realistic Evaluation Metrics
  * Combine Model-driven and Data-driven (e.g., deep unroll)
  * Generalize to real-world rains (e.g., semi/unsupervised learning, domain adaptation, transfer learning)
  * Fast, small (simple), robust video deraining
  * Deraining as Task-specific preprocessing (helps high-level tasks)
  * Multi-task learning (e.g., with snow, haze)
  * Solve Over-derain, Under-derain, and residual artifacts


<!--   * Use unpaired training data with natural (i.e., real-world) rains -->


# Models (Deraining)

## ICLR 2023 (under review)

* Selective Frequency Network for Image Restoration [[OpenReview](https://openreview.net/forum?id=tyZ1ChGZIKO)] 

* Networks are Slacking Off: Understanding Generalization Problem in Image Deraining [[OpenReview](https://openreview.net/forum?id=qGuU8To1y7x)] 

## NeurlPS 2022

* Generative Status Estimation and Information Decoupling for Image Rain Removal [[OpenReview](https://openreview.net/forum?id=C2o5DeL_8L1)]

## ECCV 2022

* Rethinking Video Rain Streak Removal: A New Synthesis Model and A Deraining Network with Video Rain Prior [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790556.pdf)] [[code](https://github.com/wangshauitj/RDD-Net)]

* Not Just Streaks: Towards Ground Truth for Single Image Deraining [[paper](https://arxiv.org/abs/2206.10779)] [[code](https://github.com/UCLA-VMG/GT-RAIN)]

* ART-SS: An Adaptive Rejection Technique for Semi-Supervised Restoration for Adverse Weather-Affected Images [[paper](https://arxiv.org/abs/2203.09275)] [[code](https://github.com/rajeevyasarla/ART-SS)] 

* Blind Image Decomposition [[paper](https://arxiv.org/abs/2108.11364)] [[code](https://github.com/JunlinHan/BID)]

## CVPR 2022

* Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.html)] [[code](https://github.com/yuyi-sd/robust_rain_removal)]

* Dreaming To Prune Image Deraining Networks [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.html)] **(NO CODE)**

* Unsupervised Deraining: Where Contrastive Learning Meets Self-Similarity [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.html)] [[code](https://github.com/yunguo224/NLCL)]

* Unpaired Deep Image Deraining Using Dual Contrastive Learning [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Unpaired_Deep_Image_Deraining_Using_Dual_Contrastive_Learning_CVPR_2022_paper.html)] **(NO CODE)**

* Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.html)] [[code](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal)]

* Neural Compression-Based Feature Learning for Video Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Neural_Compression-Based_Feature_Learning_for_Video_Restoration_CVPR_2022_paper.pdf)] **(NO CODE)**

## ICCV 2021

* Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.html)]

* Structure-Preserving Deraining with Residue Channel Prior Guidance [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.html)]

* Improving De-raining Generalization via Neural Reorganization [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.html)]


## CVPR 2021

* Self-Aligned Video Deraining With Transmission-Depth Consistency [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.html)]

* Semi-Supervised Video Deraining With Dynamical Rain Generator [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.html)]

* Robust Representation Learning with Feedback for Single Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.html)]

* From Rain Generation to Rain Removal [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_From_Rain_Generation_to_Rain_Removal_CVPR_2021_paper.html)]

* Image De-raining via Continual Learning [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.html)]

* Multi-Stage Progressive Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)]

* Multi-Decoding Deraining Network and Quasi-Sparsity Based Training [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.html)]

* Memory Oriented Transfer Learning for Semi-Supervised Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.html)]

# Datasets

* Synthetic Dataset

* Real Dataset

# Metrics

* Full-Reference
  * PSNR
  * SSIM
  * VIF
  * FSIM
  * TODO: ad others

* Non-Reference
  * NIQE
  * BRISQUE
  * SSEQ

# Repositories

* [[DerainZoo (Single Image vs. Video Based)](https://github.com/nnUyi/DerainZoo)]

* [[Video-and-Single-Image-Deraining](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]
