# Latent Diffusion Model for Quantitative Precipitation Estimation and Forecast at km Scale
Accurate high-resolution precipitation estimation remains a significant challenge in weather prediction due to computational limitations and sub-grid process parameterization difficulties. We present a latent diffusion modeling (LDM) framework that estimates 4 km resolution precipitation using 25 km resolution atmospheric and topographic inputs. The LDM transforms precipitation data into a compact Quasi-Gaussian latent space and progressively refines predictions through neural network-guided diffusion, effectively avoiding common deep learning issues such as mode collapse and blurry artifacts. 
## Model Architecture
The LDM framework consists of three main components: a Latent Encoding Model (LEM), a Conditional U-Net (c U-Net), and a Diffusion Model. 
The LEM transforms the precipitation data into latent space representations, the c U-Net leverages the input data to regress these latent representations, and the Diffusion Model quantifies the uncertainty and corrects the bias within the c U-Net-generated latent representations to generate high-resolution precipitation estimates. Figure 2 illustrates the overall model structure of the LDM. 
The c U-Net is inserted before the diffusion model to bridge the gap between the conditional information and the noise, making it easier for the model to learn to remove the noise. 
![image](https://github.com/user-attachments/assets/3dd33137-6cb0-4be0-a7ef-6c9f021bbd47)

## REstimation for Extreme events
![image](https://github.com/user-attachments/assets/d53eb6ea-3727-48d8-814d-264a93a0e866)
