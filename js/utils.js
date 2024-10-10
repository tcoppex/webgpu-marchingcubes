// ----------------------------------------------------------------------------

export function nextPowerOfTwo(x) {
  let power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function degToRad(x) {
  return x * (Math.PI / 180);
}

export function alignTo(byteLength, byteAlignment) {
  return parseInt(Math.ceil(byteLength / byteAlignment) * byteAlignment);
}

export function alignTo256(byteLength) {
  return alignTo(byteLength, 256);
}

export function getWorkgroupCount(gridSize, workgroupSize) {
  return workgroupSize.map((x, i) => parseInt(
    (gridSize[i] + x - 1) / x
  ));
}

// ----------------------------------------------------------------------------

export function setComputePass(commandEncoder, passInfo, extraBindGroups=[]) {
  const pass = commandEncoder.beginComputePass({label: passInfo.label});
  pass.setPipeline(passInfo.pipeline);
  pass.setBindGroup(0, passInfo.bindGroup);
  for (let i in extraBindGroups) {
    const desc = extraBindGroups[i];
    pass.setBindGroup(desc.index, desc.bindGroup, desc.dynamicOffsets);
  }
  if (passInfo.indirect !== undefined) {
    pass.dispatchWorkgroupsIndirect(passInfo.indirect.buffer, passInfo.indirect.offset || 0);
  } else {
    pass.dispatchWorkgroups(
      passInfo.workgroupCount[0],
      passInfo.workgroupCount[1],
      passInfo.workgroupCount[2]
    );
  }
  pass.end();  
}

export async function createStorage(device, bytesize, label='', extraUsage = 0) {
  console.assert(bytesize > 0);

  const buffer = device.createBuffer({
    label,
    size: bytesize,
    usage: GPUBufferUsage.STORAGE
         | GPUBufferUsage.COPY_DST
         | extraUsage
         ,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).fill(0);
  buffer.unmap();
  return buffer;
}

export async function createAndUploadStorage(device, data, label='') {
  console.assert(data instanceof Uint32Array); 

  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE
         | GPUBufferUsage.COPY_DST
         ,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

export async function createIndirectStorage(device, data, count, label='') {
  console.assert(data instanceof Uint32Array); 

  const indirectBuffer = device.createBuffer({
    label,
    size: count * data.byteLength,
    usage: GPUBufferUsage.STORAGE
         | GPUBufferUsage.COPY_DST
         | GPUBufferUsage.INDIRECT
         ,
    mappedAtCreation: true,
  });

  const mappedArray = new Uint32Array(indirectBuffer.getMappedRange());
  mappedArray.fill(1); //

  for (let i = 0; i < count; i++) {
    for (let j = 0; j < data.length; j++) {
      mappedArray[i*data.length + j] = data[j];
    }
  }

  indirectBuffer.unmap();
  return indirectBuffer;
}

export function resetStorage(device, buffer) {
  const zeroData = new Uint8Array(buffer.size);
  device.queue.writeBuffer(
    buffer, 
    0, 
    zeroData.buffer,
    zeroData.byteOffset,
    zeroData.byteLength
  );
}

// ----------------------------------------------------------------------------

export function createTexture(device, imageBitmap, imageFormat = 'rgba8unorm') {
  const mipLevelCount = Math.floor(Math.log2(Math.max(imageBitmap.width,imageBitmap.height))) + 1;
  const textureDescriptor = {
    size: {
      width: imageBitmap.width,
      height: imageBitmap.height,
    },
    format: imageFormat,
    usage: GPUTextureUsage.TEXTURE_BINDING
         | GPUTextureUsage.COPY_DST
         | GPUTextureUsage.RENDER_ATTACHMENT
         ,
    mipLevelCount: mipLevelCount,
  };
  const texture = device.createTexture(textureDescriptor);
  device.queue.copyExternalImageToTexture(
    {source: imageBitmap},
    {texture},
    textureDescriptor.size
  );

  // [not very efficient as generateMipmap recreate a pipeline everytime]
  if (mipLevelCount > 1) {
    generateMipmap(device, texture, textureDescriptor);
  }

  return texture;
}

export async function fetchTextureFromURL(device, url) {
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);
  return createTexture(device, imageBitmap);
}

export function generateMipmap(device, texture, textureDescriptor) {
  const shaderModule = device.createShaderModule({
    code: `
    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) texCoord: vec2f,
    };

    @vertex
    fn vs_main(
      @builtin(vertex_index) vertexIndex: u32
    ) -> VertexOutput {
      var out: VertexOutput;
      let uv = vec2f(f32((vertexIndex << 1) & 2), f32(vertexIndex & 2));
      out.position = vec4f(2.0 * uv - 1.0, 0.0, 1.0);
      out.texCoord = vec2f(uv.x, 1.0 - uv.y);
      return out;
    }

    @group(0) @binding(0) var uSampler: sampler;
    @group(0) @binding(1) var uTexture: texture_2d<f32>;

    @fragment
    fn fs_main(
      @location(0) texCoord: vec2f
    ) -> @location(0) vec4f {
      return textureSample(uTexture, uSampler, texCoord);
    }
    `
  });

  const label = 'MipMap Generator';

  const bindGroupLayout = device.createBindGroupLayout({
    label: label,
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      sampler: {},
    }, {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      texture: {},
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: label,
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{
        format: textureDescriptor.format
      }],
    },
    primitive: {
      topology: 'triangle-strip',
      stripIndexFromat: 'uint32',
    },
  });

  const sampler = device.createSampler({minFilter: 'linear'});

  const commandEncoder = device.createCommandEncoder({});

  let srcView = texture.createView({
    baseMipLevel: 0,
    mipLevelCount: 1,
  });
  for (let lvl = 1; lvl < textureDescriptor.mipLevelCount; ++lvl) {
    const dstView = texture.createView({
      baseMipLevel: lvl,
      mipLevelCount: 1,
    });
    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: dstView,
        loadOp: 'clear',
        storeOp: 'store',
      }]
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: sampler,
      }, {
        binding: 1,
        resource: srcView,
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();

    srcView = dstView;
  }
  device.queue.submit([commandEncoder.finish()]);
}

// ----------------------------------------------------------------------------

export function clearTexture(device, texture, dimension='3', format='u') {
  const workgroupSize = 4;

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'Utils::clearTexture',
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {
        access: 'write-only',
        format: texture.format,
        viewDimension: texture.dimension,
      }
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const clearPipeline = device.createComputePipeline({
    layout: pipelineLayout,
      compute: {
        module: device.createShaderModule({
          code: `
            @group(0) @binding(0) var storageTex : texture_storage_${dimension}d<${texture.format}, write>;

            @compute @workgroup_size(${workgroupSize}, ${workgroupSize}, ${workgroupSize})
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
              let texSize: vec3u = textureDimensions(storageTex);
              if (any(global_id >= texSize)) {
                return;
              }
              textureStore(storageTex, global_id, vec4${format}(0));
            }
          `
        }),
        entryPoint: "main"
      }
  });

  const clearBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: texture.createView(),
      }
    ]
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(clearPipeline);
  passEncoder.setBindGroup(0, clearBindGroup);

  const dispatchSizeX = Math.ceil(texture.width / workgroupSize);
  const dispatchSizeY = Math.ceil(texture.height / workgroupSize);
  const dispatchSizeZ = Math.ceil(texture.depthOrArrayLayers / workgroupSize);
  passEncoder.dispatchWorkgroups(dispatchSizeX, dispatchSizeY, dispatchSizeZ);

  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

// ----------------------------------------------------------------------------

export async function debug_displayBuffer(device, inBuffer, label='', type='u32') {
  console.assert(inBuffer !== undefined);

  const stagingBuffer = device.createBuffer({
    size: inBuffer.size,
    usage: GPUBufferUsage.MAP_READ
         | GPUBufferUsage.COPY_DST,
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    inBuffer, 0,
    stagingBuffer, 0,
    inBuffer.size
  );
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  
  let data = null;

  if (type==='u32') {
    data = new Uint32Array(stagingBuffer.getMappedRange());
  } else {
    data = new Float32Array(stagingBuffer.getMappedRange());
  }

  if (data.length > 512 && data.length < 2048) {
    console.log(`${label} - partial:`, 
      data.slice(0, 4), 
      data.slice(510, 514),
      data.slice(1022, 1026), 
      data.slice(1534, 1538), 
    );
  } else {
    console.log(`${label}:`, data);
  }
  stagingBuffer.unmap();
}

export async function debug_ReadTexture(device, texture) {
  console.assert(texture !== undefined);

  const textureRes = texture.width; // assumes X = Y = Z

  const layerSize = textureRes * textureRes;
  const bytesPerRow = alignTo256(textureRes * 4 * Float32Array.BYTES_PER_ELEMENT); //
  const bufferByteSize = layerSize * bytesPerRow;

  const readBuffer = device.createBuffer({
    size: bufferByteSize, //
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();

  // Copy texture to buffer
  commandEncoder.copyTextureToBuffer(
    { texture },
    {
      buffer: readBuffer,
      bytesPerRow, 
      rowsPerImage: textureRes 
    },
    {
      width: texture.width,
      height: texture.height,
      depthOrArrayLayers: texture.depthOrArrayLayers,
    }
  );

  device.queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const densityData = new Float32Array(readBuffer.getMappedRange());

  // -------------------------------------------
  // Print some values from the middle slice.
  const stride = bytesPerRow / Float32Array.BYTES_PER_ELEMENT;
  const offset = parseInt(textureRes/2) * textureRes * stride;
  for (let j = 0; j < textureRes; ++j) {
    let s = '';
    for (let i = 0; i < textureRes; i += 4) {
      const index = stride * j + i;
      const val = densityData[offset + index];
      s += ` ${val}`;
    }
    console.log(s);
  }
  // -------------------------------------------

  readBuffer.unmap();
}

// ----------------------------------------------------------------------------
