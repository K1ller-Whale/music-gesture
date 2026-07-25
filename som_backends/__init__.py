"""Optional adapters that wrap the *official* PWC-Net and I3D repos so they can
be used as drop-in backbones for the SoM motion branch (the ``external`` path in
configs/som_paper_faithful.yaml -> model.motion.flow_impl / i3d_impl).

These are TEMPLATES: they contain the correct wrapping structure and the known
gotchas (input padding for PWC-Net, 2-channel trajectory stem for I3D, spatial
feature-map extraction), but you must point them at the cloned repos and verify
the exact class/method names against the version you cloned.

Why adapters at all? The DDT only depends on two contracts:
  * flow:  module(f1, f2)  -> [B, 2, h, w]        (any h,w; DDT resizes)
  * i3d :  module.extract_features(x[B,2,T,H,W]) -> [B, C, T', H', W']
           and an int attribute ``out_channels``.
As long as an adapter satisfies those, fusion/FiLM/audio are untouched.
"""
