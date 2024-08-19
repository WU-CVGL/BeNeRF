<h1 align="center">BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream</h1>
<p align="center">
    <a href="https://akawincent.github.io">Wenpu Li</a><sup>1,5*</sup> &emsp;&emsp;
    <a href="https://github.com/pianwan">Pian Wan </a><sup>1,2*</sup> &emsp;&emsp;
    <a href="https://wangpeng000.github.io">Peng Wang</a><sup>1,3*</sup> &emsp;&emsp;
    <a href="https://jinghangli.github.io/">Jinghang Li</a><sup>4</sup> &emsp;&emsp;
    <a href="https://sites.google.com/view/zhouyi-joey/home">Yi Zhou</a><sup>4</sup> &emsp;&emsp;
    <a href="https://ethliup.github.io/">Peidong Liu</a><sup>1†</sup>
</p>

<p align="center">
    <sup>*</sup>equal contribution &emsp;&emsp; <sup>†</sup> denotes corresponding author.
</p>

<p align="center">
    <sup>1</sup>Westlake University &emsp;&emsp;
    <sup>2</sup>EPFL &emsp;&emsp;
    <sup>3</sup>Zhejiang University &emsp;&emsp;
    <sup>4</sup>Hunan University &emsp;&emsp; </br>
    <sup>5</sup>Guangdong University of Technology 
</p>

<hr>

<h5 align="center"> This paper was accepted by European Conference on Computer Vision (ECCV) 2024.</h5>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


<h5 align="center">


[![arXiv](https://img.shields.io/badge/Arxiv-2407.02174-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.02174)
[![pdf](https://img.shields.io/badge/PDF-Paper-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_paper.pdf) 
[![pdf](https://img.shields.io/badge/PDF-Supplementary-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_supplementary.pdf) 
[![Home Page](https://img.shields.io/badge/GitHubPages-ProjectPage-blue.svg?logo=GitHubPages)](https://akawincent.github.io/BeNeRF/) 
[![Dataset](https://img.shields.io/badge/OneDrive-Dataset-green.svg?logo=ProtonDrive)](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EjZNs8MwoXBDqT61v_j5V3EBIoKb8dG9KlYtYmLxcNJG_Q?e=AFXeUB)
![GitHub Repo stars](https://img.shields.io/github/stars/WU-CVGL/BeNeRF)



</h5>

<div align="center">
This repository is an official PyTorch implementation of the paper "BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream". We explore the possibility of recovering the neural radiance fields and camera motion trajectory from a single blurry image. This allows BeNeRF to decode the underlying sharp video from a single blurred image.
</div>

## 📢 News
- `2024.08.20` Training Code and datasets have been released. 
- `2024.07.01` Our paper was accepted by ECCV2024!! Thanks to all collaborators!!

## 📋 Overview

![pipeline](./doc/pipeline.jpg)

<div>
Given a single blurry image and its corresponding event stream, BeNeRF recovers the underlying 3D scene representation and the camera motion trajectory jointly. In particular, we represent the 3D scene with neural radiance fields and the camera motion trajectory with a cubic B-Spline in SE(3) space. Both the blurry image and accumulated events within a time interval can thus be synthesized from the 3D scene representation providing the camera poses. The camera trajectory, NeRF, are then optimized by minimizing the difference between the synthesized data and the real measurements.
</div>

## Result



## Quickstart

## Citation

## Acknowledgment

The overall framework, metrics computing and camera transformation are derived from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/), [BAD-NeRF](https://github.com/WU-CVGL/BAD-NeRF) respectively. We appreciate the effort of the contributors to these repositories.
