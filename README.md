# PINNs-for-Point-Set-Registration


# PINNs-for-Point-Set-Registration  <!-- 项目标题 -->

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2408.xxxxx-B31B1B)](https://arxiv.org/abs/2408.xxxxx)
[![Model](https://img.shields.io/badge/Pretrained-Model-Google%20Drive-4285F4)](https://drive.google.com/file/d/16ZNGCIL4RsnQH0xj3ZJ2c6ozL9UDoC82/view)

> **Physics-Informed Neural Networks for Biomechanics-Aware Point-Set Registration**  
> *Lin Zhaoxi, et al.*  
> IEEE TMI 2025 (to appear)

---

## 🌟 Highlights
* **Four biomechanical PINN variants**: Stress, Strain, Stress-Strain, and Deformation prediction.  
* Robust on **simulation** and **clinical** MR–TRUS prostate datasets.  
* Deformation-prediction PINN cuts RMSE by **-87 %** on simulation data and achieves **4.9 mm TRE** clinically while nearly eliminating foldings.  

---

## 🔧 Framework  
<p align="center">
  <a href="Image/architectures.pdf"><img src="Image/architectures.png" width="100%"></a>
</p>
<p align="center"><i>Fig. 1  Four FPT-PINN configurations predicting different physical quantities.</i></p>

---

## 📈 Qualitative Results  
<p align="center">
  <a href="Image/figure2.pdf"><img src="Image/figure2.png" width="100%"></a>
</p>
<p align="center"><i>Fig. 2  Surface overlap comparison on simulation (top) and clinical (bottom) cases.</i></p>

<p align="center">
  <a href="Image/figure3.pdf"><img src="Image/figure3.png" width="100%"></a>
</p>
<p align="center"><i>Fig. 3  Jacobian-determinant maps; smoother J≈1 indicates more physical deformations.</i></p>

---

## 📊 Quantitative Results  

| Dataset | Method | RMSE ↓ / TRE ↓ | CD ↓ | %neg J ↓ |
|---------|--------|---------------|------|----------|
| **Sim** | **Deformation (Ours)** | **0.219 ± 0.057 mm** | **0.194 ± 0.057 mm** | **0 %** |
|         | Stress-Strain (Ours)   | 0.242 ± 0.147 | 0.217 ± 0.098 | 0 % |
|         | Stress                 | 0.264 ± 0.098 | 0.240 ± 0.147 | 0 % |
|         | Strain (Ours)          | 0.305 ± 0.116 | 0.273 ± 0.116 | 0.004 % |
|         | WarpPINN               | 0.298 ± 0.095 | 0.269 ± 0.095 | 0 % |
|         | FPT (no PINN)          | 1.748 ± 0.912 | 1.260 ± 0.913 | **0 %** |
| **Clin.** | **Deformation (Ours)** | **4.924 ± 1.542 mm** | **2.125 ± 0.291 mm** | 0.278 % |
|          | Strain (Ours)         | 5.359 ± 1.746 | 2.129 ± 0.259 | 0.459 % |
|          | Stress                | 5.694 ± 1.780 | 2.170 ± 0.285 | 0.296 % |
|          | WarpPINN              | 5.771 ± 2.033 | 2.150 ± 0.294 | 0.506 % |
|          | FPT (no PINN)         | 5.901 ± 1.981 | 2.316 ± 0.347 | 0.281 % |

<details>
<summary>📄 Full tables & p-values</summary>

*See `tables/` folder or the supplementary PDF for complete metrics and statistical tests.*
</details>

---

## 🔥 Usage

```bash
git clone https://github.com/Msx00/PINNs-for-Point-Set-Registration.git
cd PINNs-for-Point-Set-Registration

# put the downloaded checkpoint here:
# ./log/xxxxx/checkpoints/model_best.pt

python demo_register.py --cfg configs/deformation.yaml


## Contributors

- [Zhaoxi Lin](https://github.com/Lzhaoxi)  
