This is a repository that contains computer vision algorithms that works in rainy conditions. 

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[[paper]()] [[code]()]

# Surveys

* A Comprehensive Benchmark Analysis of Single Image Deraining: Current Challenges and Future Perspectives (IJCV 2021) [[paper](https://link.springer.com/article/10.1007/s11263-020-01416-w)]
* Survey on rain removal from videos or a single image (SCI China 2021) [[paper](https://link.springer.com/content/pdf/10.1007/s11432-020-3225-9.pdf?pdf=button)]
* Single image deraining: From model-based to data-driven and beyond (TPAMI 2020) [[paper](https://arxiv.org/abs/1912.07150)]
* A Survey of Single Image De-raining in 2020 (Arxiv 2020) [[paper](https://link.springer.com/chapter/10.1007/978-981-16-3945-6_75)]

* Key takeways from these surveys
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

* Restormer: Efficient Transformer for High-Resolution Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)] [[code]()]

* Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.html)] [[code](https://github.com/yuyi-sd/robust_rain_removal)]

* Dreaming To Prune Image Deraining Networks [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.html)] **(NO CODE)**

* Unsupervised Deraining: Where Contrastive Learning Meets Self-S
imilarity [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.html)] [[code](https://github.com/yunguo224/NLCL)]

* Unpaired Deep Image Deraining Using Dual Contrastive Learning [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Unpaired_Deep_Image_Deraining_Using_Dual_Contrastive_Learning_CVPR_2022_paper.html)] **(NO CODE)**

* Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.html)] [[code](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal)]

* Neural Compression-Based Feature Learning for Video Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Neural_Compression-Based_Feature_Learning_for_Video_Restoration_CVPR_2022_paper.pdf)] **(NO CODE)**


## AAAI 2022

* Online-Updated High-Order Collaborative Networks for Single Image Deraining [[paper](https://arxiv.org/abs/2202.06568)] [[code](https://github.com/supersupercong/Online-updated-High-order-Collaborative-Networks-for-Single-Image-Deraining)]

* Close the Loop: A Unified Bottom-Up and Top-Down Paradigm for Joint Image Deraining and Segmentation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20033)] **(NO CODE)**


## TIP 2022

* Feature-Aligned Video Raindrop Removal With Temporal Constraints [[paper](https://arxiv.org/abs/2205.14574)] **(NO CODE)**


## ICCV 2021

* Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.html)] [[data](https://github.com/Yueziyu/RainDirection-and-Real3000-Dataset)] **(NO CODE)**

* Structure-Preserving Deraining with Residue Channel Prior Guidance [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.html)] [[code](https://github.com/joyies/spdnet)]

* Improving De-raining Generalization via Neural Reorganization [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.html)] **(NO CODE)**

* Let’s See Clearly: Contaminant Artifact Removal for Moving Cameras [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Lets_See_Clearly_Contaminant_Artifact_Removal_for_Moving_Cameras_ICCV_2021_paper.html)] **(NO CODE)**
 

## CVPR 2021

* Pre-Trained Image Processing Transformer [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.html)] [[code](https://github.com/huawei-noah/Pretrained-IPT)]

* Self-Aligned Video Deraining With Transmission-Depth Consistency [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.html)] [[code](https://github.com/wending94/Self-Aligned-Video-Deraining-with-Transmission-Depth-Consistency)]

* Semi-Supervised Video Deraining With Dynamical Rain Generator [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.html)] [[code](https://github.com/zsyOAOA/S2VD)]

* Robust Representation Learning with Feedback for Single Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.html)] [[code](https://github.com/LI-Hao-SJTU/DerainRLNet)]

* From Rain Generation to Rain Removal [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_From_Rain_Generation_to_Rain_Removal_CVPR_2021_paper.html)] [[code](https://github.com/hongwang01/VRGNet)]

* Image De-raining via Continual Learning [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.html)] **(NO CODE)**

* Multi-Stage Progressive Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html)] [[code](https://github.com/swz30/MPRNet)]

* Multi-Decoding Deraining Network and Quasi-Sparsity Based Training [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.html)]  **(NO CODE)**

* Memory Oriented Transfer Learning for Semi-Supervised Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.html)] [[code](https://github.com/hhb072/MOSS)]


## AAAI 2021

* EfficientDeRain: Learning Pixel-Wise Dilation Filtering for High-Efficiency Single-Image Deraining [[paper](https://arxiv.org/abs/2009.09238)] [[code](https://github.com/tsingqguo/efficientderain)]

* Rain Streak Removal via Dual Graph Convolutional Network [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16224)] **(NO CODE)**


## TIP 2021

* Online Rain/Snow Removal From Surveillance Videos [[paper](https://ieeexplore.ieee.org/document/9324987)] [[code](https://github.com/MinghanLi/OTMSCSC_matlab_2020)]

* Triple-Level Model Inferred Collaborative Network Architecture for Video Deraining [[paper](https://arxiv.org/abs/2111.04459)] **(NO CODE)**

* DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking [[paper](https://arxiv.org/abs/1912.07015)] [[code](https://github.com/OaDsis/DerainCycleGAN)]

## TPAMI 2021

* Physics-based generative adversarial models for image restoration and beyond


## ECCV 2020

* Rethinking Image Deraining via Rain Streaks and Vapors

* Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding

* Wavelet-Based Dual-Branch Network for Image Demoir´eing


## CVPR 2020

* Multi-Scale Progressive Fusion Network for Single Image Deraining

* Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes

* Detail-recovery Image Deraining via Context Aggregation Networks

* All in One Bad Weather Removal Using Architectural Search

* Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence

* A Model-driven Deep Neural Network for Single Image Rain Removal


## AAAI 2020

* Towards scale-free rain streak removal via selfsupervised fractal band learning


## TIP 2020

* Conditional Variational Image Deraining


## Year<=2019 (in chronological order) 

* A Hierarchical Approach for Rain or Snow Removing in a Single Color Image (TIP 2017)

* Deep joint rain detection and removal from a single image (CVPR2017)

* Removing rain from single images via a deep detail network (CVPR2017)

* Clearing the skies: A deep network architecture for single-image rain removal (TIP 2017)

* Attentive generative adversarial network for raindrop removal from a single image (CVPR 2018)

* Density-aware Single Image De-raining using a Multi-stream Dense Network (CVPR 2018)

* Learning dual convolutional neural networks for low-level vision (CVPR 2018)

* Non-locally enhanced encoder-decoder network for single image de-raining (ACMMM 2018)

* Recurrent squeeze-and-excitation context aggregation net for single image deraining (ECCV 2018)

* Image De-raining Using a Conditional Generative Adversarial Network (TCSVT 2019)

* Erl-net: Entangled representation learning for single image de-raining (ICCV 2019)

* Uncertainty guided multi-scale residual learning-using a cycle spinning cnn for single image de-raining (CVPR 2019)

* Heavy rain image restoration: Integrating physics model and conditional adversarial learning (CVPR 2019)

* Progressive image deraining networks: A better and simpler baseline (CVPR 2019) 

* Spatial attentive single-image deraining with a high quality real rain dataset (CVPR 2019) 

* Semi-supervised transfer learning for image rain removal (CVPR 2019) 

* Lightweight pyramid networks for image deraining (TNNLS2019)

* Joint rain detection and removal from a single image with contextualized deep networks (TPAMI2019)

* Scale-free single image deraining via visibility-enhanced recurrent wavelet learning (TIP 2019)


# Datasets

* TODO: check the category later

* Synthetic Dataset
  * Rain12
  * Rain100L
  * Rain100H
  * Rain800
  * Rain1200
  * Rain1400
  * Rain12600
  * Heavy Rain
  * RainCityScapes
  * NYU-Rain
  * MPID

* Real Dataset
  * SPA Dataset


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

# Resources

* [[DerainZoo (Single Image vs. Video Based)](https://github.com/nnUyi/DerainZoo)]

* [[Video-and-Single-Image-Deraining](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]

* [[Single Image Deraining](https://paperswithcode.com/task/single-image-deraining)]
