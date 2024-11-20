# Model Architecture
The LDM framework consists of three main components: a Latent Encoding Model (LEM), a Conditional U-Net (c U-Net), and a Diffusion Model. 
The LEM transforms the precipitation data into latent space representations, the c U-Net leverages the input data to regress these latent representations, and the Diffusion Model quantifies the uncertainty and corrects the bias within the c U-Net-generated latent representations to generate high-resolution precipitation estimates. Figure 2 illustrates the overall model structure of the LDM. 
The c U-Net is inserted before the diffusion model to bridge the gap between the conditional information and the noise, making it easier for the model to learn to remove the noise. 
![image](https://github.com/user-attachments/assets/3dd33137-6cb0-4be0-a7ef-6c9f021bbd47)

# the estimation for Extreme events
![image](https://github.com/user-attachments/assets/d53eb6ea-3727-48d8-814d-264a93a0e866)
