# ComfyUI-Fluxtapoz

A set of nodes for editing images using Flux in ComfyUI

## Examples

See `example_workflows` directory for examples.

No ControlNets are used in any of the following examples.

## Rectified Flow Inversion (Unsampling from [RF Inversion](https://rf-inversion.github.io/))

Admittedly this has some small differences between the example images in the paper, but it's very close. Will be updating as I find the issue.
It's currently my recommended way to unsample an image for editing or style transfer.

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_inversion_updated.json) for RF-Inversion.

![rf_inversion](https://github.com/user-attachments/assets/f0517649-4dbb-4371-a8d5-3ae90e3b6368)

##### Update [2024.10.16]

Stylization now works!
![rf_inversion_stylization](https://github.com/user-attachments/assets/015825b8-9253-4270-a183-610c1420ae0d)

It can also be used to mix or style images (although I'm still working out the settings for this)
![rf_inverse_mix](https://github.com/user-attachments/assets/2588fab7-3de6-4708-b1da-6da4c8be4edb)

### Node Parameters

#### Outverse Flux Model Pred Node

-   Ensure "reverse_ode" is set to True on the "Outverse Flux Model Pred" node. Sometimes when users upgrade this repo it doesn't load the workflow correctly.

#### Flux Reverse ODE Sampler

-   latent_image -- the image to guide the sampling
-   start_step -- the step that the sampler starts guiding the sampling towards the image in "latent_image"
-   end_step -- the last step for guiding the sampling (not inclusive)
-   eta -- the strength of the guidance. The paper does not decrease this below 0.7
-   eta_trend -- how the eta should increase/decrease/stay constant between start_step and end_step

#### Flux Forward ODE Sampler

-   gamma -- the paper leaves this at 0.5

#### Guidance Suggestions

-   For sampling normal flux guidance works (~3.5)
-   For unsampling use 0

#### Common Issues

-   Overlayed images -- try changing your start step and/or Eta. A start step that is too late won't be able to influence the image generation properly
-   Not following edits -- try fewer steps (change start/end step) or lower eta
-   Make sure your steps on the Forward (unsampling) and Reverse (sampling) samplers are the same (recommended 28 each)

## Other Inversion Techniques

### Inverse Noise (unsampling via DDIM)

![unsampling_example](https://github.com/user-attachments/assets/9c604a31-5cc9-49c2-9a08-98e7872591c2)

### Inject Inversed Noise

See example workflow for how to use this one. It's similar to inverse noise/unsampling, but has better adherence to the input image.

![inject_inversed_noise_example](https://github.com/user-attachments/assets/ee052855-12c6-47f7-8178-b4acfb2ca6b9)
![inject_unsampled_noise_cowboy](https://github.com/user-attachments/assets/4d92c591-e04d-4123-a432-d859a32e5f46)

## Acknowledgements

[RF-Inversion](https://rf-inversion.github.io/)

```
@article{rout2024rfinversion,
  title={Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations},
  author={Litu Rout and Yujia Chen and Nataniel Ruiz and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
  journal={arXiv preprint arXiv:2410.10792},
  year={2024}
}
```

[RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)
```
@article{wang2024taming,
  title={Taming Rectified Flow for Inversion and Editing},
  author={Wang, Jiangshan and Pu, Junfu and Qi, Zhongang and Guo, Jiayi and Ma, Yue and Huang, Nisha and Chen, Yuxin and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2411.04746},
  year={2024}
}
```
