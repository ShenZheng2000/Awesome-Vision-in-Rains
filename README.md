This is a repository that contains computer vision algorithms that works in rainy conditions. 

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[[paper]()] [[code]()] [cite=] (up till 12/20/2022)

# Surveys

* A Comprehensive Benchmark Analysis of Single Image Deraining: Current Challenges and Future Perspectives (IJCV 2021) [[paper](https://link.springer.com/article/10.1007/s11263-020-01416-w)]
* Survey on rain removal from videos or a single image (SCI China 2021) [[paper](https://link.springer.com/content/pdf/10.1007/s11432-020-3225-9.pdf?pdf=button)]
* Single image deraining: From model-based to data-driven and beyond (TPAMI 2020) [[paper](https://arxiv.org/pdf/1912.07150.pdf)]
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

* Generative Status Estimation and Information Decoupling for Image Rain Removal 
  * SEIDNet [[OpenReview](https://openreview.net/forum?id=C2o5DeL_8L1)]


## ECCV 2022

* Rethinking Video Rain Streak Removal: A New Synthesis Model and A Deraining Network with Video Rain Prior 
  * RDD-Net [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790556.pdf)] [[PyTorch](https://github.com/wangshauitj/RDD-Net)] [cite=0]

* Not Just Streaks: Towards Ground Truth for Single Image Deraining 
  * GT-RAIN [[paper](https://arxiv.org/pdf/2206.10779.pdf)] [[PyTorch](https://github.com/UCLA-VMG/GT-RAIN)] [cite=0]

* ART-SS: An Adaptive Rejection Technique for Semi-Supervised Restoration for Adverse Weather-Affected Images 
  * ART-SS [[paper](https://arxiv.org/pdf/2203.09275.pdf)] [[PyTorch](https://github.com/rajeevyasarla/ART-SS)] [cite=0]

* Blind Image Decomposition 
  * BIDeN [[paper](https://arxiv.org/pdf/2108.11364.pdf)] [[PyTorch](https://github.com/JunlinHan/BID)] [cite=0]


## CVPR 2022

* MAXIM: Multi-Axis MLP for Image Processing 
  * MAXIM [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf)] [[JAX](https://github.com/google-research/maxim)] [cite=39]

* Restormer: Efficient Transformer for High-Resolution Image Restoration 
  * Restormer [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)] [[PyTorch](https://github.com/swz30/Restormer)] [cite=179]

* Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond 
  * Yu et al. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.pdf)] [[PyTorch](https://github.com/yuyi-sd/robust_rain_removal)] [cite=4]

* Dreaming To Prune Image Deraining Networks 
  * Zou et al. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf)] **(NO CODE)** [cite=3]

* Unsupervised Deraining: Where Contrastive Learning Meets Self-Similarity 
  * NLCL [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.pdf)] [[PyTorch](https://github.com/yunguo224/NLCL)] [cite=1]

* Unpaired Deep Image Deraining Using Dual Contrastive Learning 
  * DCDGAN [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Unpaired_Deep_Image_Deraining_Using_Dual_Contrastive_Learning_CVPR_2022_paper.pdf)] **(NO CODE)** [cite=8]

* Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model 
  * Chen et al. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf)] [[PyTorch](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal)] [cite=8]

* Neural Compression-Based Feature Learning for Video Restoration
  * Huang et al. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Neural_Compression-Based_Feature_Learning_for_Video_Restoration_CVPR_2022_paper.pdf)] **(NO CODE)** [cite=3]


## AAAI 2022

* Online-Updated High-Order Collaborative Networks for Single Image Deraining 
  *  Wang et al. [[paper](https://arxiv.org/pdf/2202.06568.pdf)] [[PyTorch](https://github.com/supersupercong/Online-updated-High-order-Collaborative-Networks-for-Single-Image-Deraining)] [cite=2]

* Close the Loop: A Unified Bottom-Up and Top-Down Paradigm for Joint Image Deraining and Segmentation 
  * UBCN [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20033)] **(NO CODE)** [cite=4]


## TIP 2022

* Feature-Aligned Video Raindrop Removal With Temporal Constraints 
  * Yan et al. [[paper](https://arxiv.org/pdf/2205.14574.pdf)] **(NO CODE)** [cite=1]

## WACV 2022

* Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM
  * Li et al. [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Li_Single_Image_Deraining_Network_With_Rain_Embedding_Consistency_and_Layered_WACV_2022_paper.pdf)] [[PyTorch](https://github.com/Yizhou-Li-CV/ECNet)] [cite=6]

* FLUID: Few-Shot Self-Supervised Image Deraining 
  * Rai et al. [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Nandan_FLUID_Few-Shot_Self-Supervised_Image_Deraining_WACV_2022_paper.pdf)] **(NO CODE)** 

* SAPNet: Segmentation-Aware Progressive Network for Perceptual Contrastive Deraining 
  * Zheng et al. [[paper](https://openaccess.thecvf.com/content/WACV2022W/VAQ/papers/Zheng_SAPNet_Segmentation-Aware_Progressive_Network_for_Perceptual_Contrastive_Deraining_WACVW_2022_paper.pdf)] [[PyTorch](https://github.com/ShenZheng2000/SAPNet-for-image-deraining)] [cite=9]


## ICCV 2021

* Unpaired Learning for Deep Image Deraining with Rain Direction Regularizer 
  * UDRDR [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.pdf)] [[data](https://github.com/Yueziyu/RainDirection-and-Real3000-Dataset)] **(NO CODE)** [cite=10]

* Structure-Preserving Deraining with Residue Channel Prior Guidance 
  * SPDNet [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Structure-Preserving_Deraining_With_Residue_Channel_Prior_Guidance_ICCV_2021_paper.pdf)] [[PyTorch](https://github.com/joyies/spdnet)] [cite=19]

* Improving De-raining Generalization via Neural Reorganization 
  * NR [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.pdf)] **(NO CODE)** [cite=4]

* Let’s See Clearly: Contaminant Artifact Removal for Moving Cameras 
  * Li et al. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Lets_See_Clearly_Contaminant_Artifact_Removal_for_Moving_Cameras_ICCV_2021_paper.pdf)] **(NO CODE)** [cite=6]
 

## CVPR 2021

* Pre-Trained Image Processing Transformer 
  * IPT [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/huawei-noah/Pretrained-IPT)] [cite=615]

* Self-Aligned Video Deraining With Transmission-Depth Consistency 
  * Yan et al. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Self-Aligned_Video_Deraining_With_Transmission-Depth_Consistency_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/wending94/Self-Aligned-Video-Deraining-with-Transmission-Depth-Consistency)] [cite=8]

* Semi-Supervised Video Deraining With Dynamical Rain Generator 
  * S2VD [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/zsyOAOA/S2VD)] [cite=26]

* Robust Representation Learning with Feedback for Single Image Deraining 
  * DerainRLNet  [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Robust_Representation_Learning_With_Feedback_for_Single_Image_Deraining_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/LI-Hao-SJTU/DerainRLNet)] [cite=35]

* From Rain Generation to Rain Removal 
  * VRGNet [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Rain_Generation_to_Rain_Removal_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/hongwang01/VRGNet)] [cite=32]

* Image De-raining via Continual Learning 
  * IDCL [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)] **(NO CODE)** [cite=15]

* Multi-Stage Progressive Image Restoration 
  * MPRNet [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/swz30/MPRNet)] [cite=432]

* Multi-Decoding Deraining Network and Quasi-Sparsity Based Training 
  * MDDNet [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multi-Decoding_Deraining_Network_and_Quasi-Sparsity_Based_Training_CVPR_2021_paper.pdf)]  **(NO CODE)** [cite=11]

* Memory Oriented Transfer Learning for Semi-Supervised Image Deraining 
  * MOSS [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/hhb072/MOSS)] [cite=37]

* HINet: Half Instance Normalization Network for Image Restoration 
  * HINet [[paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_HINet_Half_Instance_Normalization_Network_for_Image_Restoration_CVPRW_2021_paper.pdf)] [[PyTorch](https://github.com/megvii-model/HINet)] [cite=107]


## AAAI 2021

* EfficientDeRain: Learning Pixel-Wise Dilation Filtering for High-Efficiency Single-Image Deraining 
  * EfficientDeRain [[paper](https://arxiv.org/pdf/2009.09238.pdf)] [[PyTorch](https://github.com/tsingqguo/efficientderain)] [cite=33]

* Rain Streak Removal via Dual Graph Convolutional Network 
  * DualGCN [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16224)] **(NO CODE)** [cite=43]


## TIP 2021

* Online Rain/Snow Removal From Surveillance Videos 
  * OMS-CSC [[paper](https://ieeexplore.ieee.org/document/9324987)] [[MATLAB](https://github.com/MinghanLi/OTMSCSC_matlab_2020)] [cite=20]

* Triple-Level Model Inferred Collaborative Network Architecture for Video Deraining 
  * TMICS [[paper](https://arxiv.org/pdf/2111.04459.pdf)] [[PyTorch](https://github.com/dut-media-lab/TMICS)] [cite=4]

* DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking 
  * DerainCycleGAN [[paper](https://arxiv.org/pdf/1912.07015.pdf)] [[PyTorch](https://github.com/OaDsis/DerainCycleGAN)] [cite=76]

## TPAMI 2021

* Physics-based generative adversarial models for image restoration and beyond 
  * PBGAN [[paper](https://ieeexplore.ieee.org/document/8968618)] [[PyTorch](https://github.com/cuiyixin555/PhysicsGan)] [cite=97]


## Year==2020

* Rethinking Image Deraining via Rain Streaks and Vapors (ECCV 2020) 
  * Wang et al. [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620358.pdf)] [[PyTorch](https://github.com/yluestc/derain)] [cite=33]

* Beyond Monocular Deraining: Stereo Image Deraining via Semantic Understanding (ECCV 2020) 
  * PRRNet [[paper]([https://paperswithcode.com/paper/beyond-monocular-deraining-stereo-image](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720069.pdf))] **(NO CODE)** [cite=25]

* Wavelet-Based Dual-Branch Network for Image Demoireing (ECCV 2020) 
  * WDNet [[paper]([https://arxiv.org/pdf/2007.07173.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580086.pdf))] **(NO CODE)** [cite=42]

* Multi-Scale Progressive Fusion Network for Single Image Deraining (CVPR 2020) 
  * MSPFN [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Multi-Scale_Progressive_Fusion_Network_for_Single_Image_Deraining_CVPR_2020_paper.pdf)] [[TensorFlow](https://github.com/kuijiang94/MSPFN)] [cite=263]

* Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes (CVPR 2020) 
  * Syn2Real [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yasarla_Syn2Real_Transfer_Learning_for_Image_Deraining_Using_Gaussian_Processes_CVPR_2020_paper.pdf)] [[PyTorch](https://github.com/rajeevyasarla/Syn2Real)] [cite=115]

* Detail-recovery Image Deraining via Context Aggregation Networks (CVPR 2020) 
  * DRD-Net [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf)] [[Keras](https://github.com/Dengsgithub/DRD-Net)] [cite=110]

* All in One Bad Weather Removal Using Architectural Search (CVPR 2020) 
  * AIONet [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf)] **(NO CODE)** [cite=65]

* Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence (CVPR 2020) 
  * SLDNet [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf)] [[PyTorch](https://github.com/flyywh/CVPR-2020-Self-Rain-Removal)] [cite=38]

* A Model-driven Deep Neural Network for Single Image Rain Removal (CVPR 2020) 
  * RCDNet [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)] [[PyTorch](https://github.com/hongwang01/RCDNet)] [cite=161]

* Towards scale-free rain streak removal via selfsupervised fractal band learning (AAAI 2020) 
  * FBL [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6954)] [[PyTorch](https://github.com/flyywh/AAAI-2020-FBL-SS)] [cite=25]

* Conditional Variational Image Deraining (TIP 2020) 
  * CVID [[paper](https://arxiv.org/pdf/2004.11373v2.pdf)] [[TensorFlow](https://github.com/Yingjun-Du/VID)] [cite=47]


## Year==2019
 
* Image De-raining Using a Conditional Generative Adversarial Network (TCSVT 2019) 
  * ID-CGAN [[paper](https://arxiv.org/pdf/1701.05957v4.pdf)] [[Lua](https://github.com/hezhangsprinter/ID-CGAN)] [cite=800]

* Erl-net: Entangled representation learning for single image de-raining (ICCV 2019) 
  * Erl-net [[paper](https://paperswithcode.com/paper/erl-net-entangled-representation-learning-for)] **(NO CODE)** [cite=60]

* RainFlow: Optical Flow Under Rain Streaks and Rain Veiling Effect (ICCV 2019) 
  * RainFlow [[paper](https://paperswithcode.com/paper/rainflow-optical-flow-under-rain-streaks-and)] **(NO CODE)** [cite=25]

* Physics-Based Rendering for Improving Robustness to Rain (CVPR 2019) 
  * Halder et al. [[paper](https://paperswithcode.com/paper/physics-based-rendering-for-improving)] **(NO CODE)** [cite=72]

* HeavyRainRestorer: Integrating physics model and conditional adversarial learning (CVPR 2019) 
  * Heavy rain image restoration [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Heavy_Rain_Image_Restoration_Integrating_Physics_Model_and_Conditional_Adversarial_CVPR_2019_paper.pdf)] [[PyTorch](https://github.com/liruoteng/HeavyRainRemoval)] [cite=219]

* Progressive image deraining networks: A better and simpler baseline (CVPR 2019) 
  * PreNet [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.pdf)] [[PyTorch](https://github.com/csdwren/PReNet)] [cite=470]

* Spatial attentive single-image deraining with a high quality real rain dataset (CVPR 2019) 
  * SPANet [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_CVPR_2019_paper.pdf)] [[PyTorch](https://github.com/stevewongv/SPANet)] [cite=313]

* Semi-supervised transfer learning for image rain removal (CVPR 2019) 
  * SEMI [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Semi-Supervised_Transfer_Learning_for_Image_Rain_Removal_CVPR_2019_paper.pdf)] [[TensorFlow](https://github.com/wwzjer/Semi-supervised-IRR)] [cite=219]

* Depth-attentional Features for Single-image Rain Removal (CVPR 2019) 
  * DAF-Net [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)] [[Caffe](https://github.com/xw-hu/DAF-Net)] [cite=199]

* Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining (CVPR 2019) 
  * UMRL [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)] [[PyTorch](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)] [cite=154]

* Frame-Consistent Recurrent Video Deraining With Dual-Level Flow (CVPR 2019) 
  * Yang et al. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Frame-Consistent_Recurrent_Video_Deraining_With_Dual-Level_Flow_CVPR_2019_paper.pdf)] **(NO CODE)** [cite=74]

* Singe Image Rain Removal with Unpaired Information: A Differentiable Programming Perspective (AAAI 2019) 
  * RR-GAN [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4971)] **(NO CODE)** [cite=45]

* Lightweight pyramid networks for image deraining (TNNLS2019) 
  * LPNet [[paper](https://arxiv.org/pdf/1805.06173v1.pdf)] [[TensorFlow](https://xueyangfu.github.io/projects/LPNet.html)] [cite=224]

* Joint rain detection and removal from a single image with contextualized deep networks (TPAMI2019) 
  * JORDER-E [[paper](https://ieeexplore.ieee.org/document/8627954)] [[Pytorch](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal)] [cite=221]

* Scale-free single image deraining via visibility-enhanced recurrent wavelet learning (TIP 2019) 
  * Yang et al. [[paper](https://ieeexplore.ieee.org/document/8610325)] [[Pytorch](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal)] [cite=82]


## Year == 2018

* Attentive generative adversarial network for raindrop removal from a single image (CVPR 2018) 
  * Attentive GAN [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)] [[PyTorch](https://github.com/rui1996/DeRaindrop)] [cite=474]

* Density-aware Single Image De-raining using a Multi-stream Dense Network (CVPR 2018) 
  * DID-MDN [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.pdf)] [[PyTorch](https://github.com/hezhangsprinter/DID-MDN)] [cite=623]

* Learning dual convolutional neural networks for low-level vision (CVPR 2018) 
  * DualCNN  [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pan_Learning_Dual_Convolutional_CVPR_2018_paper.pdf)] [[PyTorch](https://github.com/jspan/dualcnn)] [cite=151]

* Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos (CVPR 2018) 
  * J4R-Net [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)] [[MATLAB](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)] [cite=152]

* Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework (CVPR 2018) 
  * SPAC-CNN [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Robust_Video_Content_CVPR_2018_paper.pdf)] [[MATLAB](https://bitbucket.org/st_ntu_corplab/mrp2a/src/bd2633dbc9912b833de156c799fdeb82747c1240/?at=master)] [cite=136]

* Video Rain Streak Removal by Multiscale Convolutional Sparse Coding (CVPR 2018) 
  * MSCSC [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Video_Rain_Streak_CVPR_2018_paper.pdf)] [[MATLAB](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)] [cite=160]

* Non-locally enhanced encoder-decoder network for single image de-raining (ACMMM 2018) 
  * NLEDN [[paper](https://arxiv.org/pdf/1808.01491v1.pdf)] [[PyTorch](https://github.com/AlexHex7/NLEDN)] [cite=195]

* Recurrent squeeze-and-excitation context aggregation net for single image deraining (ECCV 2018) 
  * RESCAN [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.pdf)] [[PyTorch](https://github.com/XiaLiPKU/RESCAN)] [cite=514]


## Year == 2017

* A Hierarchical Approach for Rain or Snow Removing in a Single Color Image (TIP 2017) 
  * Wang et al. [[paper](https://ieeexplore.ieee.org/document/7934435)] **(NO CODE)** [cite=139]

* Should We Encode Rain Streaks in Video as Deterministic or Stochastic? (ICCV 2017) 
  * MoG [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wei_Should_We_Encode_ICCV_2017_paper.pdf)] [[MATLAB](https://github.com/wwzjer/RainRemoval_ICCV2017)] [cite=126]

* Joint Bi-Layer Optimization for Single-Image Rain Streak Removal (ICCV 2017) 
  * JBO [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.pdf)] **(NO CODE)** [cite=247]

* Deep joint rain detection and removal from a single image (CVPR2017) 
  * JORDER [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] [[MATLAB](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] [cite=685]

* Removing rain from single images via a deep detail network (CVPR2017) 
  * DDN [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[TensorFlow](https://xueyangfu.github.io/projects/cvpr2017.html)] [cite=742]

* Video Desnowing and Deraining Based on Matrix Decomposition (CVPR2017) 
  * Ren et al. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Video_Desnowing_and_CVPR_2017_paper.pdf)] **(NO CODE)** [cite=146]
 
* A Novel Tensor-Based Video Rain Streaks Removal Approach via Utilizing Discriminatively Intrinsic Priors (CVPR 2017) 
  * FastDeRain [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.pdf)] **(NO CODE)** [cite=150]

* Clearing the skies: A deep network architecture for single-image rain removal (TIP 2017) 
  * Clearing The Skies [[paper](https://arxiv.org/pdf/1609.02087v2.pdf)] [[TensorFlow](https://xueyangfu.github.io/projects/tip2017.html)] [cite=618]



# Datasets


* Synthetic Dataset
  * Rain12 [[Layer Priors](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Li_Rain_Streak_Removal_CVPR_2016_paper.pdf)] [[link](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg#list/path=%2Fsharelink3792638399-290876125944720%2Fdatasets&parentPath=%2Fsharelink3792638399-290876125944720)]
    * same link for Rain100L, Rain100H, Rain1400/Rain12600
  * Rain100L [[JORDER](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] 
  * Rain100H [[JORDER](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] 
  * Rain1400/Rain12600 [[DDN](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)]
  * Rain800 [[ID-CGAN](https://arxiv.org/pdf/1701.05957.pdf)] [[link](https://github.com/hezhangsprinter/ID-CGAN)] 
  * Rain12000 [[DID-MDN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.pdf)]  [[link](https://github.com/hezhangsprinter/DID-MDN)]  
  * Rain14000 [[DDN](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[link](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3Q)]
  * Outdoor-Rain [[HeavyRainRestorer](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Heavy_Rain_Image_Restoration_Integrating_Physics_Model_and_Conditional_Adversarial_CVPR_2019_paper.pdf)]  [[link](https://www.dropbox.com/sh/zpadllquvmaztib/AACmzqQmGrRMp7qqXjbb7Gfza?dl=0)]  
  * RainCityScapes [[DAF-Net](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)] [[link](https://github.com/xw-hu/DAF-Net)] 
  * NYU-Rain [[HeavyRainRestorer](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Heavy_Rain_Image_Restoration_Integrating_Physics_Model_and_Conditional_Adversarial_CVPR_2019_paper.pdf)] [[link](https://github.com/liruoteng/HeavyRainRemoval)]  
  * NTURain [[SPAC-CNN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Robust_Video_Content_CVPR_2018_paper.pdf)] [[link](https://github.com/hotndy/SPAC-SupplementaryMaterials)]
  * RainMotion [[RDD-Net](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790556.pdf)] [[link](https://drive.google.com/file/d/1905B_e2RgQGnyfHd5xpjB4lTLYoq0Jm4/view)]
  * BID [[BIDeN](https://arxiv.org/pdf/2108.11364.pdf)] [[link](https://drive.google.com/drive/folders/1wUUKTiRAGVvelarhsjmZZ_1iBdBaM6Ka)]
  * RainSynLight25 & RainSynComplex25 [[J4R-Net](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)] [[link](https://drive.google.com/file/d/1uFir819r0gkcDWbBWvAIhXsU7JdLm-6t/view)]

* Real-World Dataset
  * Raindrop [[Attentive GAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.pdf)] [[link](https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K)]
  * DDN-SIRR [[SEMI](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Semi-Supervised_Transfer_Learning_for_Image_Rain_Removal_CVPR_2019_paper.pdf)] [[link](https://github.com/wwzjer/Semi-supervised-IRR/tree/master/data/rainy_image_dataset/real_input)]
  * MOSS [[MOSS](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.pdf)] [[link](https://github.com/hhb072/MOSS/tree/main/data/real/input)]
  * SPA-Data [[SPANet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_CVPR_2019_paper.pdf)] [[link](https://pan.baidu.com/s/1lPn3MWckHxh1uBYYucoWVQ)]
    * password:4fwo
  * GT-RAIN [[GT-RAIN](https://arxiv.org/pdf/2206.10779.pdf)] [[link](https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8)]

* Task-Driven Dataset
  * MPID [[Li Survey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Single_Image_Deraining_A_Comprehensive_Benchmark_Analysis_CVPR_2019_paper.pdf)] [[link](https://github.com/panda-lab/Single-Image-Deraining)]  
 
# Metrics

* Full-Reference
  * PSNR [[code](https://github.com/aizvorski/video-quality/blob/master/psnr.py)]
  * SSIM [[code](https://github.com/aizvorski/video-quality/blob/master/ssim.py)]
  * VIF [[code](https://github.com/aizvorski/video-quality/blob/master/vifp.py)]
  * FSIM
  * UQI

* Non-Reference
  * NIQE [[code](https://github.com/aizvorski/video-quality/blob/master/niqe.py)]
  * UNIQUE [[code](https://github.com/zwx8981/UNIQUE)]
  * SPAQ [[code](https://github.com/h4nwei/SPAQ)]
  * BRISQUE
  * SSEQ

* Task-driven 
  * mAP
  * mPA
  * mIoU

# Resources

* [[DerainZoo (Single Image vs. Video Based)](https://github.com/nnUyi/DerainZoo)]

* [[Video-and-Single-Image-Deraining](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]

* [[Single Image Deraining](https://paperswithcode.com/task/single-image-deraining)]
