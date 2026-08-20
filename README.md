# DreamHand

**Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand
Motion Recovery**

<p align="center">
<a href="https://ggxxii.github.io/dreamhand/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
</p>

<p align="center"><img src="assets/teaser.png" width="92%"></p>

Egocentric video offers scalable manipulation data for embodied AI, yet
recovering metric 3D hand trajectories remains challenging due to severe object
occlusion and frequent out-of-sight gaps. Existing single-frame and windowed
temporal regressors fail when hands leave the image, while recent video
diffusion models rely on heavy, stochastic multi-step sampling as pixel-space
renderers. We instead repurpose the VDM into a deterministic geometry encoder.
A single forward pass over its clean latent exposes scene content beyond
current observations, including occluded and out-of-sight hands. We introduce
DreamHand, an offline clip-level framework that extracts features via a
Deterministic Clean-Latent Encoder and decodes them with a Bidirectional
Spatiotemporal Decoder. It recovers continuous bimanual trajectories with
metric placement and no external detector, while a Ray-Based Camera Solver
renders test-time camera intrinsics optional. Across five egocentric
benchmarks, DreamHand sets a new state of the art, cutting MPJPE by 30% on
occlusion-heavy ARCTIC and 40% on HOT3D. These gains reach 46%–61% when
evaluating out-of-sight hands, offering a scalable path from everyday human
video to robot manipulation data.

<p align="center"><img src="assets/method.png" width="92%"></p>

## TODO

- [ ] arXiv preprint
- [ ] Project page
- [ ] Inference code
- [ ] Pretrained checkpoints — DreamHand (K-given) and DreamHand (K-free)
- [ ] Training code and data preparation scripts

The code and the pretrained checkpoints are not released yet. They will be
published in this repository; watch it for updates.

## Citation

```bibtex
@article{dreamhand2026,
  title  = {DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust
            Egocentric 3D Hand Motion Recovery},
  author = {},
  year   = {2026},
}
```

## License and acknowledgements

The code will be released under the MIT License (see `LICENSE`). The Wan
backbone weights, the VideoX-Fun framework, the MANO hand model and the
training datasets are third-party assets under their own licences. This work
builds on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) /
[Wan](https://github.com/Wan-Video/Wan2.2) and
[smplx](https://github.com/vchoutas/smplx).
