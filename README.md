# webgpu-marchingcubes

Dynamic marching cubes mesh generation via signed distant function in WebGPU. 

### Quickstart

To run locally you'll need to generate a PEM encoded SSL certificate and private key, then run an https server with those certificates:

```bash
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout localhost-key.pem \
  -out localhost.pem \
  -days 365 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"


http-server -S -C localhost.pem -K localhost-key.pem -p 8443
```

### Input Controls

* Mouse right click to rotate.
* Mouse wheel to dolly.

## Acknowledgment

Texture assets courtesy of [Poly Haven](https://polyhaven.com/) under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## References

* Geiss, Ryan. "[Generating Complex Procedural Terrains Using the GPU.](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu)" In GPU Gems 3, edited by Hubert Nguyen, 7–37. NVIDIA Corporation, 2007. 
* Quílez, Íñigo. "[Distance Functions.](https://iquilezles.org/articles/distfunctions/)" Inigo Quilez Blog, 2011. https://iquilezles.org/articles/distfunctions/.
* McEwan, Ian, David Sheets, Stefan Gustavson, and Mark Richardson. "Efficient Computational Noise in GLSL." CoRR abs/1204.1461 (2012). https://arxiv.org/abs/1204.1461
* Gustavson, Stefan. "webgl-noise" GitHub repository, 2012. [https://github.com/stegu/webgl-noise/](https://github.com/stegu/webgl-noise/).
