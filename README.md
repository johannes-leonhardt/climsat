# ClimSat - A Diffusion Autoencoder Model for Climate-conditional Satellite Image Editing

This repository contains the code for our paper "ClimSat - A Diffusion Autoencoder Model for Climate-conditional Satellite Image Editing" published in Science of Remote Sensing in 2025.

## Reference

Leonhardt, Johannes, Jürgen Gall, and Ribana Roscher. "ClimSat–A diffusion autoencoder model for climate-conditional satellite image editing." Science of Remote Sensing (2025): 100235. https://doi.org/10.1016/j.srs.2025.100235.

## Abstract

Climatic conditions have a strong impact on the Earth's surface, especially in terms of how different land cover classes appear and the way they are distributed. Satellite images are valuable data for studying these effects. However, disentangling the specific influence of climate remains a complex challenge. This paper proposes ClimSat, an image editing model designed to realistically simulate prescribed climate conditions on satellite imagery. The proposed ClimSat model is constructed as a diffusion autoencoder, and it incorporates contextual information through multi-conditional batch normalization and classifier-free guidance. The technical capabilities of ClimSat were first validated by demonstrating its ability to generate high-quality images which remain faithful to the prescribed conditions. The experimental results further show that ClimSat outperforms other models in terms of both criteria. ClimSat’s practical utility is demonstrated in two downstream applications, i.e., data augmentation for land cover classification, where training on ClimSat-augmented datasets improves classifier generalizability beyond regionally limited datasets, and climate change visualization, where the effects of forecast climate change are simulated under two socioeconomic pathways for protected regions in Finland and Italy.
