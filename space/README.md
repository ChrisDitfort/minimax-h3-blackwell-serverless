---
title: Privora H3 RunPod Worker
emoji: "🔒"
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Privora H3 RunPod Worker

Private, model-free Docker build source for the PrivoraVideo MiniMax H3 RunPod Serverless
worker. This Space is an image build/registry boundary, not a public application UI.

The image contains the validated `multimodal-4` application runtime but no H3 weight
payloads. At RunPod startup it requires the exact cached snapshot of
`CDitfort/privora-minimax-h3-models` at
`ecb69a4211d74b5798398021003bccde02d63757` and refuses mutable refs or runtime downloads.

Publishing this Space does not deploy or modify a RunPod endpoint.
