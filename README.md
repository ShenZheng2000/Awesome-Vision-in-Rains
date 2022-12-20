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

* MAXIM: Multi-Axis MLP for Image Processing [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf)] [[code](https://github.com/google-research/maxim)]

* Restormer: Efficient Transformer for High-Resolution Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)] [[code](https://github.com/swz30/Restormer)]

* Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.pdf)] [[code](https://github.com/yuyi-sd/robust_rain_removal)]

* Dreaming To Prune Image Deraining Networks [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf)] **(NO CODE)**

* Unsupervised Deraining: Where Contrastive Learning Meets Self-S
imilarity [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.pdf)] [[code](https://github.com/yunguo224/NLCL)]

* Unpaired Deep Image Deraining Using Dual Contrastive Learning [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Unpaired_Deep_Image_Deraining_Using_Dual_Contrastive_Learning_CVPR_2022_paper.pdf)] **(NO CODE)**

* Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf)] [[code](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal)]

* Neural Compression-Based Feature Learning for Video Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Neural_Compression-Based_Feature_Learning_for_Video_Restoration_CVPR_2022_paper.pdf)] **(NO CODE)**


## AAAI 2022

* Online-Updated High-Order Collaborative Networks for Single Image Deraining [[paper](https://arxiv.org/abs/2202.06568)] [[code](https://github.com/supersupercong/Online-updated-High-order-Collaborative-Networks-for-Single-Image-Deraining)]

* Close the Loop: A Unified Bottom-Up and Top-Down Paradigm for Joint Image Deraining and Segmentation [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20033)] **(NO CODE)**


## TIP 2022

* Feature-Aligned Video Raindrop Removal With Temporal Constraints [[paper](https://arxiv.org/abs/2205.14574)] **(NO CODE)**


## ICCV 2021

* Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.pdf)] [[data](https://github.com/Yueziyu/RainDirection-and-Real3000-Dataset)] **(NO CODE)**

* Structure-Preserving Deraining with Residue Channel Prior Guidance [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.pdf)] [[code](https://github.com/joyies/spdnet)]

* Improving De-raining Generalization via Neural Reorganization [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.pdf)] **(NO CODE)**

* Let’s See Clearly: Contaminant Artifact Removal for Moving Cameras [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Lets_See_Clearly_Contaminant_Artifact_Removal_for_Moving_Cameras_ICCV_2021_paper.pdf)] **(NO CODE)**
 

## CVPR 2021

* Pre-Trained Image Processing Transformer [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)] [[code](https://github.com/huawei-noah/Pretrained-IPT)]

* Self-Aligned Video Deraining With Transmission-Depth Consistency [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.pdf)] [[code](https://github.com/wending94/Self-Aligned-Video-Deraining-with-Transmission-Depth-Consistency)]

* Semi-Supervised Video Deraining With Dynamical Rain Generator [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf)] [[code](https://github.com/zsyOAOA/S2VD)]

* Robust Representation Learning with Feedback for Single Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.pdf)] [[code](https://github.com/LI-Hao-SJTU/DerainRLNet)]

* From Rain Generation to Rain Removal [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Rain_Generation_to_Rain_Removal_CVPR_2021_paper.pdf)] [[code](https://github.com/hongwang01/VRGNet)]

* Image De-raining via Continual Learning [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)] **(NO CODE)**

* Multi-Stage Progressive Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf)] [[code](https://github.com/swz30/MPRNet)]

* Multi-Decoding Deraining Network and Quasi-Sparsity Based Training [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.pdf)]  **(NO CODE)**

* Memory Oriented Transfer Learning for Semi-Supervised Image Deraining [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.pdf)] [[code](https://github.com/hhb072/MOSS)]\

* HINet: Half Instance Normalization Network for Image Restoration [[paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_HINet_Half_Instance_Normalization_Network_for_Image_Restoration_CVPRW_2021_paper.pdf)] [[code](https://github.com/megvii-model/HINet)]


## AAAI 2021

* EfficientDeRain: Learning Pixel-Wise Dilation Filtering for High-Efficiency Single-Image Deraining [[paper](https://arxiv.org/abs/2009.09238)] [[code](https://github.com/tsingqguo/efficientderain)]

* Rain Streak Removal via Dual Graph Convolutional Network [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16224)] **(NO CODE)**


## TIP 2021

* Online Rain/Snow Removal From Surveillance Videos [[paper](https://ieeexplore.ieee.org/document/9324987)] [[code](https://github.com/MinghanLi/OTMSCSC_matlab_2020)]

* Triple-Level Model Inferred Collaborative Network Architecture for Video Deraining [[paper](https://arxiv.org/abs/2111.04459)] **(NO CODE)**

* DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking [[paper](https://arxiv.org/abs/1912.07015)] [[code](https://github.com/OaDsis/DerainCycleGAN)]

## TPAMI 2021

* Physics-based generative adversarial models for image restoration and beyond [[paper](https://ieeexplore.ieee.org/document/8968618)] [[code](https://github.com/cuiyixin555/PhysicsGan)]


## Year==2020

* Rethinking Image Deraining via Rain Streaks and Vapors (ECCV 2020) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620358.pdf)] [[code](https://github.com/yluestc/derain)]

* Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding (ECCV 2020) [[paper]([https://paperswithcode.com/paper/beyond-monocular-deraining-stereo-image](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf))] **(NO CODE)**

* Wavelet-Based Dual-Branch Network for Image Demoireing (ECCV 2020) [[paper]([https://arxiv.org/abs/2007.07173](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580086.pdf))] **(NO CODE)**

* Multi-Scale Progressive Fusion Network for Single Image Deraining (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Multi-Scale_Progressive_Fusion_Network_for_Single_Image_Deraining_CVPR_2020_paper.pdf)] [[code](https://github.com/kuijiang94/MSPFN)]

* Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yasarla_Syn2Real_Transfer_Learning_for_Image_Deraining_Using_Gaussian_Processes_CVPR_2020_paper.pdf)] [[code](https://github.com/rajeevyasarla/Syn2Real)]

* Detail-recovery Image Deraining via Context Aggregation Networks (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf)] [[code](https://github.com/Dengsgithub/DRD-Net)]

* All in One Bad Weather Removal Using Architectural Search (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf)] **(NO CODE)**

* Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf)] [[code](https://github.com/flyywh/CVPR-2020-Self-Rain-Removal)]

* A Model-driven Deep Neural Network for Single Image Rain Removal (CVPR 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)] [[code](https://github.com/hongwang01/RCDNet)]

* Towards scale-free rain streak removal via selfsupervised fractal band learning (AAAI 2020) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6954)] **(NO CODE)**

* Conditional Variational Image Deraining (TIP 2020) [[paper](https://arxiv.org/pdf/2004.11373v2.pdf)] [[code](https://github.com/Yingjun-Du/VID)]


## Year==2019
 
* Image De-raining Using a Conditional Generative Adversarial Network (TCSVT 2019) [[paper](https://paperswithcode.com/paper/image-de-raining-using-a-conditional)] [[code](https://arxiv.org/pdf/1701.05957v4.pdf)]

* Erl-net: Entangled representation learning for single image de-raining (ICCV 2019) [[paper](https://paperswithcode.com/paper/erl-net-entangled-representation-learning-for)] [[code](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)]

* RainFlow: Optical Flow Under Rain Streaks and Rain Veiling Effect (ICCV 2019) [[paper](https://paperswithcode.com/paper/rainflow-optical-flow-under-rain-streaks-and)] **(NO CODE)**

* Physics-Based Rendering for Improving Robustness to Rain (CVPR 2019) [[paper](https://paperswithcode.com/paper/physics-based-rendering-for-improving)] **(NO CODE)**

* Heavy rain image restoration: Integrating physics model and conditional adversarial learning (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Heavy_Rain_Image_Restoration_Integrating_Physics_Model_and_Conditional_Adversarial_CVPR_2019_paper.pdf)] [[code](https://github.com/liruoteng/HeavyRainRemoval)]

* Progressive image deraining networks: A better and simpler baseline (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.pdf)] [[code](https://github.com/csdwren/PReNet)]

* Spatial attentive single-image deraining with a high quality real rain dataset (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_CVPR_2019_paper.pdf)] [[code](https://github.com/stevewongv/SPANet)]

* Semi-supervised transfer learning for image rain removal (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Semi-Supervised_Transfer_Learning_for_Image_Rain_Removal_CVPR_2019_paper.pdf)] [[code](https://github.com/wwzjer/Semi-supervised-IRR)]

* Depth-attentional Features for Single-image Rain Removal (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)] [[code](https://github.com/xw-hu/DAF-Net)]

* Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)] [[code](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)]

* Frame-Consistent Recurrent Video Deraining With Dual-Level Flow (CVPR 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Frame-Consistent_Recurrent_Video_Deraining_With_Dual-Level_Flow_CVPR_2019_paper.pdf)] [[code](https://github.com/flyywh/Dual-FLow-Video-Deraining-CVPR-2019)]

* Singe Image Rain Removal with Unpaired Information (AAAI 2019) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4971)] **(NO CODE)**

* Lightweight pyramid networks for image deraining (TNNLS2019) [[paper](https://arxiv.org/pdf/1805.06173v1.pdf)] [[code](https://xueyangfu.github.io/projects/LPNet.html)]

* Joint rain detection and removal from a single image with contextualized deep networks (TPAMI2019) [[paper](https://ieeexplore.ieee.org/document/8627954)] [[code]()]

* Scale-free single image deraining via visibility-enhanced recurrent wavelet learning (TIP 2019) [[paper]()] [[code](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal)]


## Year == 2018

* Attentive generative adversarial network for raindrop removal from a single image (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)] [[code](https://github.com/rui1996/DeRaindrop)]

* Density-aware Single Image De-raining using a Multi-stream Dense Network (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.pdf)] [[code](https://github.com/hezhangsprinter/DID-MDN)]

* Learning dual convolutional neural networks for low-level vision (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pan_Learning_Dual_Convolutional_CVPR_2018_paper.pdf)] [[code](https://github.com/jspan/dualcnn)]

* Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)] [[code](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)]

* Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Robust_Video_Content_CVPR_2018_paper.pdf)] [[code](https://bitbucket.org/st_ntu_corplab/mrp2a/src/bd2633dbc9912b833de156c799fdeb82747c1240/?at=master)]

* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding (CVPR 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)] [[code](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)]

* Non-locally enhanced encoder-decoder network for single image de-raining (ACMMM 2018) [[paper](https://arxiv.org/pdf/1808.01491v1.pdf)] [[code](https://github.com/AlexHex7/NLEDN)]

* Recurrent squeeze-and-excitation context aggregation net for single image deraining (ECCV 2018) [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.pdf)] [[code](https://github.com/XiaLiPKU/RESCAN)]


## Year == 2017

* A Hierarchical Approach for Rain or Snow Removing in a Single Color Image (TIP 2017) [[paper](https://ieeexplore.ieee.org/document/7934435)] **(NO CODE)**

* Should We Encode Rain Streaks in Video as Deterministic or Stochastic? (ICCV 2017) [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)] [[code](https://github.com/wwzjer/RainRemoval_ICCV2017)]

* Joint Bi-Layer Optimization for Single-Image Rain Streak Removal (ICCV 2017) [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.pdf)] **(NO CODE)**

* Deep joint rain detection and removal from a single image (CVPR2017) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] [[code](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]

* Removing rain from single images via a deep detail network (CVPR2017) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[code](https://xueyangfu.github.io/projects/cvpr2017.html)]

* Video Desnowing and Deraining Based on Matrix Decomposition (CVPR2017) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Video_Desnowing_and_CVPR_2017_paper.pdf)] **(NO CODE)**
 
* A Novel Tensor-Based Video Rain Streaks Removal Approach via Utilizing Discriminatively Intrinsic Priors (CVPR 2017) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.pdf)] **(NO CODE)**

* Clearing the skies: A deep network architecture for single-image rain removal (TIP 2017) [[paper](https://arxiv.org/pdf/1609.02087v2.pdf)] [[code](https://xueyangfu.github.io/projects/tip2017.html)]



# Datasets

* TODO: check the category later
* TODO: add more dataset later

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

* Non-Reference
  * NIQE
  * BRISQUE
  * SSEQ

# Resources

* [[DerainZoo (Single Image vs. Video Based)](https://github.com/nnUyi/DerainZoo)]

* [[Video-and-Single-Image-Deraining](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]

* [[Single Image Deraining](https://paperswithcode.com/task/single-image-deraining)]
