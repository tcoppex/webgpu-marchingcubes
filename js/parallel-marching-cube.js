// ----------------------------------------------------------------------------

import * as MarchingCubeData from './data.marching-cube.js';
import * as ShaderUtils from './shader-utils.js';
import * as Utils from './utils.js';

import PerfLogger from './PerfLogger.js';

// ----------------------------------------------------------------------------

const kChunkDim = 24;
const kVoxelsPerSlice = kChunkDim * kChunkDim;
const kVoxelsPerChunk = kVoxelsPerSlice * kChunkDim;

// We add a margin to the density texture to calculate normals & occlusion.
const kChunkMargin = 4;
const kWindowDim = kChunkDim + 2 * kChunkMargin;

const kChunkSize = 12.0; //

const kDensityVolumeTexRes = kWindowDim;
const kTexelSize = 1.0 / kDensityVolumeTexRes;

// [should be updated depending on kChunkDim, and density function complexity]
const kHeuristicChunkMaxVertices = (1 << 13); // 4096
const kHeuristicChunkMaxIndices = (1 << 16); // 32768

// ---------------------

const kHeuristicMaxNonEmptyCellsSize = kHeuristicChunkMaxVertices //
                                     * Uint32Array.BYTES_PER_ELEMENT
                                     ;

const kVertexStride = (4 + 4) * Float32Array.BYTES_PER_ELEMENT;
const kHeuristicChunkVerticesBufferSize = kHeuristicChunkMaxVertices 
                                        * kVertexStride
                                        ;

const kHeuristicChunkIndicesBufferSize = kHeuristicChunkMaxIndices
                                        * Uint32Array.BYTES_PER_ELEMENT
                                        ;

const kMaxLinearGroupSize = 256;
const kDrawIndexedIndirectSize = 5 * Uint32Array.BYTES_PER_ELEMENT;

const perf = new PerfLogger();

// ----------------------------------------------------------------------------

export class Generator {
  constructor() {
    this.device = null;
    this.buffer = null;
    this.nearestSampler = null;
    this.linearSampler = null;
    this.densityTexture = null;
    this.passInfo = null;
  }

  async init(device, density) {
    console.assert(device != null);
    console.assert(density != null);

    this.device = device;

    // ----
    this.densityBindGroupInfo = {
      index: 2,  //
      bindGroup: density.bindGroup,
      dynamicOffsets: [],
    };
    // ----

    // Output buffers.
    const [nonEmptyCells, verticesToGenerate] = await Promise.all([
      Utils.createStorage(this.device, kHeuristicMaxNonEmptyCellsSize, `nonEmptyCells`),
      Utils.createStorage(this.device, kHeuristicMaxNonEmptyCellsSize * 3, `verticesToGenerate`),
    ]);
    const [atomicCountCells, atomicCountVertices, atomicCountIndices] = await Promise.all([
      Utils.createStorage(this.device, Uint32Array.BYTES_PER_ELEMENT, `atomicCountCells`, GPUBufferUsage.COPY_SRC),
      Utils.createStorage(this.device, Uint32Array.BYTES_PER_ELEMENT, `atomicCountVertices`, GPUBufferUsage.COPY_SRC),
      Utils.createStorage(this.device, Uint32Array.BYTES_PER_ELEMENT, `atomicCountIndices`, GPUBufferUsage.COPY_SRC),
    ]);
    const [indirectCells, indirectVertices] = await Promise.all([
      Utils.createIndirectStorage(this.device, new Uint32Array([1, 1, 1]), 1, 'indirectCells'),
      Utils.createIndirectStorage(this.device, new Uint32Array([1, 1, 1]), 1, 'indirectVertices'),
    ]);

    this.stagingBuffer = this.device.createBuffer({
      label: 'stagingBuffer',
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ
           | GPUBufferUsage.COPY_DST
           ,
    });

    this.buffer = {
      nonEmptyCells,
      verticesToGenerate,

      atomicCountCells,
      atomicCountVertices,
      atomicCountIndices,

      indirectCells,
      indirectVertices,
    };

    // ---------------------------

    this.nearestSampler = this.device.createSampler({});
    this.linearSampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // [use 'rgba16float' to be able to use filtering, while only needing 'r16float']
    this.densityTexture = this.device.createTexture({
      label: `MarchingCubes::Generator::densityTexture`,
      size: {
        width: kDensityVolumeTexRes,
        height: kDensityVolumeTexRes,
        depthOrArrayLayers: kDensityVolumeTexRes,
      },
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING
           | GPUTextureUsage.TEXTURE_BINDING
           ,
    });

    // [use 3*width RED instead of RGB values to avoid concurrent texel store operations]
    // [we would use a 'r16uint' if it was compatible with storage_binding] 
    this.vertexIndicesVolume = this.device.createTexture({
      label: `MarchingCubes::Generator::vertexIndicesVolume`,
      size: {
        width: 3 * kChunkDim,
        height: kChunkDim,
        depthOrArrayLayers: kChunkDim,
      },
      dimension: '3d',
      format: 'r32uint',
      usage: GPUTextureUsage.STORAGE_BINDING
           | GPUTextureUsage.TEXTURE_BINDING
           ,
    });

    // ---------------------------

    this.chunkBindGroupLayout = this.device.createBindGroupLayout({
      label: `MarchingCubes::Generator::ChunkBindGroupLayout`,
      entries: [
      // VERTICES
      { 
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          hasDynamicOffset: true,
        }
      },
      // INDICES
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          hasDynamicOffset: true,
        }
      },
      // UNIFORM
      { 
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
        }
      },]
    });

    // ---------------------------

    this.passInfo = {
      buildDensityVolume: buildDensityVolume_PassInfo(
        this.device,
        density,
        this.chunkBindGroupLayout,
        this.densityTexture,
      ),

      listNonEmptyCells: listNonEmptyCells_PassInfo(
        this.device,
        this.nearestSampler,
        this.densityTexture,
        this.buffer.nonEmptyCells,
        this.buffer.atomicCountCells,
      ),

      listVerticesToGenerate: listVerticesToGenerate_PassInfo(
        this.device,
        this.buffer.atomicCountCells,
        this.buffer.nonEmptyCells,
        this.buffer.verticesToGenerate,
        this.buffer.atomicCountVertices,
        this.buffer.indirectCells,
      ),

      splatVertexIndices: splatVertexIndices_PassInfo(
        this.device,
        this.buffer.atomicCountVertices,
        this.buffer.verticesToGenerate,
        this.vertexIndicesVolume,
        this.buffer.indirectVertices,
      ),

      generateVertices: generateVertices_PassInfo(
        this.device,
        this.chunkBindGroupLayout,
        this.nearestSampler,
        this.linearSampler,
        this.densityTexture,
        this.buffer.atomicCountVertices,
        this.buffer.verticesToGenerate,
        this.buffer.indirectVertices,
      ),

      generateIndices: generateIndices_PassInfo(
        this.device,
        this.chunkBindGroupLayout,
        this.buffer.atomicCountCells,
        this.buffer.nonEmptyCells,
        this.vertexIndicesVolume,
        this.buffer.atomicCountIndices,
        this.buffer.indirectCells,
      ),

      setupIndirectCells: setupIndirectBuffer_PassInfo(
        this.device,
        this.buffer.atomicCountCells,
        this.buffer.indirectCells,
        kMaxLinearGroupSize,
      ),

      setupIndirectVertices: setupIndirectBuffer_PassInfo(
        this.device,
        this.buffer.atomicCountVertices,
        this.buffer.indirectVertices,
        kMaxLinearGroupSize,
      ),
    };
  }

  release() {
    Object.entries(this.buffer).foreach(([k, v]) => v.destroy());
    this.nearestSampler.destroy();
    this.linearSampler.destroy();
    this.densityTexture.destroy();
    this.vertexIndicesVolume.destroy();
    this.passInfo = null;
  }

  newGrid(dimension) {
    console.assert(dimension !== undefined);
    return new Grid(this, dimension);
  }

  buildChunk(chunk, drawIndexedIndirectBuffer = null) {
    // [might be improved by using a reset compute kernel]
    {
      Utils.resetStorage(this.device, this.buffer.atomicCountCells); 
      Utils.resetStorage(this.device, this.buffer.atomicCountVertices);
      Utils.resetStorage(this.device, this.buffer.atomicCountIndices);

      // [debug]
      // Utils.resetStorage(this.device, this.buffer.nonEmptyCells); //
      // Utils.resetStorage(this.device, this.buffer.verticesToGenerate); //
      Utils.clearTexture(this.device, this.vertexIndicesVolume); //
    }

    const extraBindGroups = [chunk.bindGroupInfo, this.densityBindGroupInfo];
    const commandEncoder = this.device.createCommandEncoder({label: 'MarchingCubes::Generator::CommandEncoder'});
    {
      Utils.setComputePass(commandEncoder, this.passInfo.buildDensityVolume, extraBindGroups);
      
      Utils.setComputePass(commandEncoder, this.passInfo.listNonEmptyCells);
      Utils.setComputePass(commandEncoder, this.passInfo.setupIndirectCells);

      Utils.setComputePass(commandEncoder, this.passInfo.listVerticesToGenerate);
      Utils.setComputePass(commandEncoder, this.passInfo.setupIndirectVertices);

      Utils.setComputePass(commandEncoder, this.passInfo.splatVertexIndices);

      Utils.setComputePass(commandEncoder, this.passInfo.generateVertices, extraBindGroups);
      Utils.setComputePass(commandEncoder, this.passInfo.generateIndices, extraBindGroups);

      if (drawIndexedIndirectBuffer) {
        commandEncoder.copyBufferToBuffer(
          this.buffer.atomicCountIndices, 0, drawIndexedIndirectBuffer,
          chunk.offsets.drawIndexedIndirect, this.buffer.atomicCountIndices.size
        );
      }

      // ----------

      // [optional, retrieve data for debugging]
      commandEncoder.copyBufferToBuffer(
        this.buffer.atomicCountCells, 0, this.stagingBuffer, 0*Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT
      );
      commandEncoder.copyBufferToBuffer(
        this.buffer.atomicCountVertices, 0, this.stagingBuffer, 1*Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT
      );
      commandEncoder.copyBufferToBuffer(
        this.buffer.atomicCountIndices, 0, this.stagingBuffer, 2*Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT
      );
    }
    this.device.queue.submit([commandEncoder.finish()]);
  }

  async DEBUG_readStaging() {
    // await this.stagingBuffer.mapAsync(GPUMapMode.READ);
    // let values = new Uint32Array(this.stagingBuffer.getMappedRange()); 
    // let nCells = values[0];
    // let nVerts = values[1];
    // let nIndex = values[2];
    // this.stagingBuffer.unmap();
    // console.log(`cells: ${nCells}, vertices: ${nVerts}, indices: ${nIndex}, `);
  }
};

// ----------------------------------------------------------------------------

class Grid {
  constructor(generator, dimension) {
    this.generator = generator;
    this.buffer = null;
    this.reset(dimension);
  }

  reset(dimension) {
    const [X, Y, Z] = dimension;

    this.size = X * Y * Z;
    this.dimension = dimension;
    this.chunks = new Array(this.size).fill(null);
    this.vertexBufferStride = kHeuristicChunkVerticesBufferSize;
    this.indexBufferStride = kHeuristicChunkIndicesBufferSize;
    this.uniformBufferStride = Utils.alignTo256(4 * Float32Array.BYTES_PER_ELEMENT); //

    const startPosition = dimension.map(d => - 0.5 * d);

    let index = 0;
    for (let k = 0; k < Z; ++k) {
      for (let j = 0; j < Y; ++j) {
        for (let i = 0; i < X; ++i) {
          const coords = [i, j, k];
          const coordsWS = coords.map((d, idx) => kChunkSize * (startPosition[idx] + d));
          const offsets = {
            vertex: index * this.vertexBufferStride,
            index: index * this.indexBufferStride,
            uniform: index * this.uniformBufferStride,
            drawIndexedIndirect: index * kDrawIndexedIndirectSize,
          };
          this.chunks[index] = new Chunk(index, coords, coordsWS, offsets);
          ++index;
        }
      }
    }

    if (this.buffer != null) {
      release();
    }
  }

  async init() {
    const [vertices, indices, drawIndexedIndirect] = await Promise.all([
      Utils.createStorage(this.generator.device, this.size * this.vertexBufferStride, `vertices`, GPUBufferUsage.VERTEX),
      Utils.createStorage(this.generator.device, this.size * this.indexBufferStride, `indices`, GPUBufferUsage.INDEX),
      Utils.createIndirectStorage(this.generator.device, new Uint32Array([0, 1, 0, 0, 0]), this.size, 'drawIndexedIndirect'),
    ]);

    const uniforms = this.generator.device.createBuffer({
      size: this.size * this.uniformBufferStride,
      usage: GPUBufferUsage.COPY_DST
           | GPUBufferUsage.UNIFORM
           ,
      mappedAtCreation: true,
    });
    const attributes = new Float32Array(uniforms.getMappedRange());
    for (let i = 0; i < this.size; ++i) {
      const chunk = this.chunks[i];
      const index = chunk.offsets.uniform / Float32Array.BYTES_PER_ELEMENT;
      attributes[index + 0] = chunk.coordsWS[0];
      attributes[index + 1] = chunk.coordsWS[1];
      attributes[index + 2] = chunk.coordsWS[2];
      attributes[index + 3] = chunk.size;
    }
    uniforms.unmap();

    this.buffer = {
      vertices,
      indices,
      uniforms,
      drawIndexedIndirect,
    };

    this.bindGroup = this.generator.device.createBindGroup({
      label: `ChunkBindGroup`,
      layout: this.generator.chunkBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffer.vertices, offset: 0, size: this.vertexBufferStride } },
        { binding: 1, resource: { buffer: this.buffer.indices, offset: 0, size: this.indexBufferStride } },
        { binding: 2, resource: { buffer: this.buffer.uniforms, offset: 0, size: this.uniformBufferStride } },
      ]
    });
    this.chunks.forEach(c => c.setBindGroupInfo(this.bindGroup));
  }

  release() {
    Object.entries(this.buffer).foreach(([k, v]) => v.destroy());
    this.buffer = null;
    this.bindGroup.destroy();
    this.bindGroup = null;
  }

  async build() {
    if (this.buffer == null) {
      await this.init();
    }

    for (let chunk_id in this.chunks) {
      this.generator.buildChunk(this.chunks[chunk_id], this.buffer.drawIndexedIndirect);
    }
  }

  draw(passEncoder) {
    this.chunks.forEach(chunk => {
      const offsets = chunk.offsets;
      passEncoder.setVertexBuffer(0, this.buffer.vertices, offsets.vertex);
      passEncoder.setIndexBuffer(this.buffer.indices, 'uint32', offsets.index); //
      passEncoder.drawIndexedIndirect(this.buffer.drawIndexedIndirect, offsets.drawIndexedIndirect);
    });
  }
};

// ----------------------------------------------------------------------------

class Chunk {
  constructor(index, coords, coordsWS, offsets) {
    this.index = index;
    this.coords = coords;
    this.size = kChunkSize;
    this.coordsWS = coordsWS;
    this.offsets = offsets;
    this.bindGroupInfo = {} // (to be filled by generator)
  }

  setBindGroupInfo(bindGroup) {
    this.bindGroupInfo = {
      index: 1, 
      bindGroup, 
      dynamicOffsets: [
        this.offsets.vertex, 
        this.offsets.index, 
        this.offsets.uniform,
      ]
    };
  }
};

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

function buildDensityVolume_PassInfo(
  device, 
  density, 
  chunkBindGroupLayout, 
  densityTexture
) {
  const label = 'MarchingCubes::Pass::BuildDensityVolume';

  const workgroupSize = [4, 4, 4]
  const gridSize = [kDensityVolumeTexRes, kDensityVolumeTexRes, kDensityVolumeTexRes];
  const workgroupCount = Utils.getWorkgroupCount(gridSize, workgroupSize);

  const computeShaderCode = `
    override kChunkDim: f32 = f32(${kChunkDim});
    override kInvChunkDim: f32 = ${1.0 / kChunkDim};
    override kInvChunkDimMinusOne: f32 = ${1.0 / (kChunkDim - 1.0)};
    override kChunkMargin: f32 = f32(${kChunkMargin});
    override kChunkSize: f32 = f32(${kChunkSize});

    // ----

    @group(0) @binding(0) var outDensityVolume: texture_storage_3d<${densityTexture.format}, write>;
    @group(1) @binding(2) var<uniform> inChunkAttributes: vec4f;

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let coords = global_invocation_id;

      if (any(coords >= vec3u(${gridSize}))) {
        return;
      }

      let ext_coords = vec3f(coords) * vec3f(kChunkDim * kInvChunkDimMinusOne);

      // Compute chunk coordinates in world-space.
      let voxelCoords = (ext_coords - vec3f(kChunkMargin)) * kInvChunkDim;
      let voxelCoordsWS = fma(voxelCoords, vec3f(kChunkSize), inChunkAttributes.xyz);

      // Compute density function for the chunk.
      let density: f32 = computeDensity( voxelCoordsWS );

      // Store result in the volume texture.
      textureStore(outDensityVolume, coords, vec4f(density, 0.0, 0.0, 0.0));
    }

    ${density.shader}
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format: densityTexture.format,
          viewDimension: densityTexture.dimension,
        }
      }
    ]
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: densityTexture.createView({dimension: '3d'}) },
    ]
  });

  const pipelineDesc = {
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout, chunkBindGroupLayout, density.bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        code: computeShaderCode
      }),
    },
  };
  const pipeline = device.createComputePipeline(pipelineDesc);

  return {
    label,
    pipelineDesc,
    pipeline,
    bindGroup,
    workgroupCount,
  };
}


function listNonEmptyCells_PassInfo(
  device,
  nearestSampler,
  densityTexture,
  nonEmptyCellsBuffer,
  atomicCountBuffer,
) {
  const label = 'MarchingCubes::Pass::ListNonEmptyCells';

  const workgroupSize = [4, 4, 4];
  const gridSize = [kChunkDim, kChunkDim, kChunkDim]; //
  const workgroupCount = Utils.getWorkgroupCount(gridSize, workgroupSize);

  const computeShaderCode = `
    override kChunkMargin: f32 = f32(${kChunkMargin});
    override kTexelSize: f32 = ${kTexelSize};

    @group(0) @binding(0) var uSamplerNearest: sampler;
    @group(0) @binding(1) var uDensityTexture: texture_3d<f32>;
    @group(0) @binding(2) var<storage, read_write> nonEmptyCellsBuffer: array<u32>;
    @group(0) @binding(3) var<storage, read_write> atomicCount: atomic<u32>;

    fn sampleNearest(uvw: vec3f) -> f32 {
      return textureSampleLevel(uDensityTexture, uSamplerNearest, uvw, 0.0).r;
    }

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let coords: vec3u = global_invocation_id;
      let resolution: vec3u = vec3u(${gridSize});

      if (any(coords >= resolution)) {
        return;
      }

      // Global thread index.
      let gid: u32 = coords.z * (resolution.x * resolution.y) 
                   + coords.y * resolution.x 
                   + coords.x
                   ;

      // ----

      let uvw    = kTexelSize * (vec3f(coords) + vec3f(kChunkMargin));
      let offset = vec2f(kTexelSize, 0.0f);

      let sideA = vec4f(
        sampleNearest(uvw + offset.yyy),
        sampleNearest(uvw + offset.yxy),
        sampleNearest(uvw + offset.xxy),
        sampleNearest(uvw + offset.xyy)
      );
      let sideB = vec4f(
        sampleNearest(uvw + offset.yyx),
        sampleNearest(uvw + offset.yxx),
        sampleNearest(uvw + offset.xxx),
        sampleNearest(uvw + offset.xyx)
      );

      let iA = vec4u(step(vec4f(0.0), sideA));
      let iB = vec4u(step(vec4f(0.0), sideB));
      let cube_case = (iA.x << 0) | (iA.y << 1) | (iA.z << 2) | (iA.w << 3) 
                    | (iB.x << 4) | (iB.y << 5) | (iB.z << 6) | (iB.w << 7)
                    ;
      // ----

      if ((cube_case > 0) && (cube_case < 255)) 
      {
        let out_index: u32 = atomicAdd(&atomicCount, 1u);
        nonEmptyCellsBuffer[out_index] = pack4xU8(vec4u(coords, cube_case));
      }
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        sampler: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          viewDimension: '3d',
          sampleType: 'float',
          multisampled: false,
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage'
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage'
        },
      },
    ]
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: nearestSampler },
      { binding: 1, resource: densityTexture.createView({dimension: '3d'}) },
      { binding: 2, resource: { buffer: nonEmptyCellsBuffer } },
      { binding: 3, resource: { buffer: atomicCountBuffer } },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  return {
    label,
    pipeline,
    bindGroup,
    workgroupCount,
  }
}


function listVerticesToGenerate_PassInfo(
  device,
  cellCountBuffer,
  nonEmptyCellsBuffer,
  verticesToGenerateBuffer,
  atomicCountBuffer,
  indirectBuffer,
) {
  const label = 'MarchingCubes::Pass::ListVerticesToGenerate';

  const workgroupSize = [kMaxLinearGroupSize]; 

  const computeShaderCode = `
    @group(0) @binding(0) var<storage> inNumCells: u32;
    @group(0) @binding(1) var<storage> inNonEmptyCells: array<u32>;
    @group(0) @binding(2) var<storage, read_write> outVerticesToGenerate: array<u32>;
    @group(0) @binding(3) var<storage, read_write> atomicCount: atomic<u32>;

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let cell_id: u32 = global_invocation_id.x;

      if (cell_id >= inNumCells) {
        return;
      }

      let data: vec4u = unpack4xU8(inNonEmptyCells[cell_id]);
      let coords: vec3u = data.xyz;
      let cube_case: u32 = data.w;

      let bits = vec4u(
        (cube_case >> 3) & 1,
        (cube_case >> 1) & 1, 
        (cube_case >> 4) & 1,
        cube_case & 1,
      );
      let vertOnEdges: vec3<bool> = (bits.xyz != bits.www);
      
      let numVerts: u32 = u32(vertOnEdges.x) 
                        + u32(vertOnEdges.y) 
                        + u32(vertOnEdges.z) 
                        ; 
      var out_index: u32 = atomicAdd(&atomicCount, numVerts);

      let edge_nums = vec3u(3, 0, 8);
      for (var i = 0; i < 3; i += 1) {
        if (vertOnEdges[i]) {
          outVerticesToGenerate[out_index] = pack4xU8(vec4u(coords, edge_nums[i]));
          out_index += 1u;
        }
      }
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage'
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage'
        },
      },
    ]
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cellCountBuffer } },
      { binding: 1, resource: { buffer: nonEmptyCellsBuffer } },
      { binding: 2, resource: { buffer: verticesToGenerateBuffer } },
      { binding: 3, resource: { buffer: atomicCountBuffer } },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  return {
    label,
    pipeline,
    bindGroup,
    indirect: {
      buffer: indirectBuffer
    }
  }
}


function splatVertexIndices_PassInfo(
  device,
  vertexCountBuffer,
  verticesToGenerateBuffer,
  vertexIndicesVolume,
  indirectBuffer,
) {
  const label = 'MarchingCubes::Pass::SplatVertexIndices';
  const workgroupSize = [kMaxLinearGroupSize];

  const computeShaderCode = `
    @group(0) @binding(0) var<storage> inVertexCount: u32;
    @group(0) @binding(1) var<storage> inVerticesToGenerate: array<u32>;
    @group(0) @binding(2) var outVertexIndicesVolume: texture_storage_3d<${vertexIndicesVolume.format}, write>;

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let vertex_id: u32 = global_invocation_id.x;

      if (vertex_id >= inVertexCount) {
        return;
      }

      let data: vec4u = unpack4xU8(inVerticesToGenerate[vertex_id]);
      let edge: u32 = data.w & 0xf;

      // [!] Use a single component texture with 3*width instead of a RGB texture
      //     to avoid concurrent writing on the same texel.
      var coords: vec3u = data.xyz;
      coords.x = 3u * coords.x + u32(select(select(0u, 1u, edge == 0u), 2u, edge == 8u));

      textureStore(outVertexIndicesVolume, coords, vec4u(vertex_id, 0u, 0u, 0u));
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format: vertexIndicesVolume.format,
          viewDimension: vertexIndicesVolume.dimension,
        }
      },
    ]
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: vertexCountBuffer } },
      { binding: 1, resource: { buffer: verticesToGenerateBuffer } },
      { binding: 2, resource: vertexIndicesVolume.createView({dimension: '3d'}) },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  return {
    label,
    pipeline,
    bindGroup,
    indirect: {
      buffer: indirectBuffer
    }
  }
}


function setupIndirectBuffer_PassInfo(
  device,
  atomicCountBuffer,
  indirectBuffer,
  targetGroupSize
) {
  console.assert(atomicCountBuffer !== undefined);
  console.assert(indirectBuffer !== undefined);
  console.assert(targetGroupSize !== undefined);

  const label = 'MarchingCubes::Pass::SetupIndirectBuffer';
  const workgroupCount = [1, 1, 1];

  const computeShaderCode = `
    override kTargetGroupSize: u32 = u32(${targetGroupSize}); //

    @group(0) @binding(0) var<storage> inTotalCount: u32;
    @group(0) @binding(1) var<storage, read_write> outIndirectDispatch: vec3u;

    @compute @workgroup_size(1)
    fn main() {
      outIndirectDispatch.x = u32((inTotalCount + (kTargetGroupSize - 1u)) / kTargetGroupSize);
      outIndirectDispatch.y = 1u;
      outIndirectDispatch.z = 1u;
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage'
        },
      },
    ]
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: atomicCountBuffer } },
      { binding: 1, resource: { buffer: indirectBuffer } },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  return {
    label,
    pipeline,
    bindGroup,
    workgroupCount,
  }
}


function generateVertices_PassInfo(
  device,
  chunkBindGroupLayout,
  nearestSampler,
  linearSampler,
  densityTexture,
  vertexCountBuffer,
  verticesToGenerateBuffer,
  indirectBuffer,
) {
  const label = 'MarchingCubes::Pass::GenerateVertices';

  const workgroupSize = [kMaxLinearGroupSize];

  const computeShaderCode = `
    override kInvWindowDim: f32 = ${1.0 / (kWindowDim)};
    override kInvWindowDimMinusOne: f32 = ${1.0 / (kWindowDim - 1)};
    override kVoxelSizeWS: f32 = ${(kChunkSize / (kChunkDim - 1.0))};
    override kTexelSize: f32 = ${kTexelSize}; //
    override kChunkMargin: f32 = f32(${kChunkMargin});

    ${MarchingCubeData.Shader_EdgesConstants}

    ${ShaderUtils.SphereRays64}

    // U(i) = pow(vec4f(1-i/16), 0.3)
    const kAOWeights = array<f32, 16>(
      1.000000000,
      0.980824675,
      0.960732353,
      0.939608660,
      0.917314755,
      0.893679531,
      0.868488366,
      0.841466359,
      0.812252396,
      0.780357156,
      0.745091108,
      0.705431757,
      0.659753955,
      0.605202038,
      0.535886731,
      0.435275282,
    );

    struct Vertex {
      position: vec3f,
      ao: f32,
      normal: vec3f,
    };

    // ------------------

    @group(0) @binding(0) var uSamplerNearest: sampler;
    @group(0) @binding(1) var uSamplerLinear: sampler;
    @group(0) @binding(2) var uDensityTexture: texture_3d<f32>;
    @group(0) @binding(3) var<storage> inNumVertices: u32;
    @group(0) @binding(4) var<storage> inVerticesToGenerateBuffer: array<u32>;

    @group(1) @binding(0) var<storage, read_write> outVertices: array<vec4f>;
    @group(1) @binding(2) var<uniform> inChunkAttributes: vec4f;

    // ------------------

    fn sampleNearest(uvw: vec3f) -> f32 {
      return textureSampleLevel(uDensityTexture, uSamplerNearest, uvw, 0.0).r;
    }

    fn sampleLinear(uvw: vec3f) -> f32 {
      return textureSampleLevel(uDensityTexture, uSamplerLinear, uvw, 0.0).r; 
    }

    fn sampleGradient(uvw: vec3f, offset: vec3f) -> f32 {
      return sampleLinear(uvw + offset) 
           - sampleLinear(uvw - offset)
           ;
    }

    fn sampleNormal(uvw: vec3f) -> vec3f {
      let offset = vec2f(kTexelSize, 0.0);
      let n = vec3f(
        sampleGradient(uvw, offset.xyy),
        sampleGradient(uvw, offset.yxy),
        sampleGradient(uvw, offset.yyx)
      );
      return normalize(n);
    }

    // [could be costly, should use a hemisphere instead of a sphere]
    fn calculateAO(uvw: vec3f, n: vec3f) -> f32 {
      var ao = 1.0;
      
      let kAORayCount: u32 = 64u;
      let kAOStepCount: u32 = 16u;
      let kRayStartOffset: f32 = 1.75;

      for (var i: u32 = 0u; i < kAORayCount; i += 1u) {
        let ray_dir: vec3f = kSphereRays64[i];
        let ray_start: vec3f = uvw;
        let ray_step: vec3f = ray_dir * (kTexelSize / f32(kAOStepCount)); //

        var ray_ao: f32 = 1.0;

        // Sample density texture.
        var ray_curr = fma(vec3f(kRayStartOffset * kTexelSize), ray_dir, ray_start);
        for (var j: u32 = 0u; j < kAOStepCount; j += 1u) {
          ray_curr += ray_step;
          let density: f32 = sampleLinear(ray_curr);
          let coeff: f32 = (4.5 * density) * kAOWeights[j];
          ray_ao = smoothstep(0.0, ray_ao, coeff);
        }

        ao += (1.0 - smoothstep(pow(ray_ao, 400.0), 1.0, dot(ray_dir, n)));

        // [we could use the density function to calculate ao furthermore]
      }
      ao = saturate(ao / f32(kAORayCount));

      return ao;
    }
    
    fn place_vertex_on_edge(coordsWS: vec3f, texcoord: vec3f, edge_id: u32) -> Vertex {
      var v: Vertex;

      let edge_start: vec3f = kEdgeStart[edge_id];
      let edge_end: vec3f = kEdgeEnd[edge_id];
      let edge_dir: vec3f = kEdgeDir[edge_id];
      let scale = vec3f(kInvWindowDimMinusOne);

      let p1 = sampleNearest(fma(edge_start, scale, texcoord));
      let p2 = sampleNearest(fma(edge_end, scale, texcoord));
      let t = vec3f(saturate(p1 / (p1 - p2)));
      let pos = fma(edge_dir, t, edge_start);
      let uvw = fma(pos, scale, texcoord);

      // World-space attributes.
      v.position = fma(vec3f(kVoxelSizeWS), pos, coordsWS);
      v.normal = sampleNormal(uvw);
      v.ao = calculateAO(uvw, v.normal);

      return v;
    }

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let vertex_id: u32 = global_invocation_id.x;

      if (vertex_id >= inNumVertices) {
        return;
      }

      let data: vec4u = unpack4xU8(inVerticesToGenerateBuffer[vertex_id]);
      let coords: vec3f = vec3f(data.xyz);
      let edge: u32 = data.w;

      let coordsWS: vec3f = fma(vec3f(kVoxelSizeWS), coords, inChunkAttributes.xyz);
      let texcoord = (coords + kChunkMargin) * kInvWindowDim;

      let v: Vertex = place_vertex_on_edge(coordsWS, texcoord, edge);

      let out_index: u32 = 2u * vertex_id;
      outVertices[out_index + 0u] = vec4f(v.position, v.ao);
      outVertices[out_index + 1u] = vec4f(v.normal, 0.0);
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        sampler: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        sampler: {},
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: 'float',
          viewDimension: '3d',
          multisampled: false,
        }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        },
      },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout, chunkBindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: nearestSampler },
      { binding: 1, resource: linearSampler },
      { binding: 2, resource: densityTexture.createView({dimension: '3d'}) },
      { binding: 3, resource: { buffer: vertexCountBuffer } },
      { binding: 4, resource: { buffer: verticesToGenerateBuffer } },
    ]
  });

  return {
    label,
    pipeline,
    bindGroup,
    indirect: {
      buffer: indirectBuffer
    }
  }
}


function generateIndices_PassInfo(
  device,
  chunkBindGroupLayout,
  cellCountBuffer,
  nonEmptyCellsBuffer,
  vertexIndicesVolume,
  atomicCountBuffer,
  indirectBuffer,
) {
  const label = 'MarchingCubes::Pass::GenerateIndices';

  const workgroupSize = [kMaxLinearGroupSize];

  const computeShaderCode = `
    ${MarchingCubeData.Shader_EdgesConstants}
    ${MarchingCubeData.Shader_kNumTriangleLUT}
    ${MarchingCubeData.Shader_kTriangleLUT}

    @group(0) @binding(0) var<storage> inNumCells: u32;
    @group(0) @binding(1) var<storage> inNonEmptyCells: array<u32>;
    @group(0) @binding(2) var inIndicesVolume: texture_3d<u32>;
    @group(0) @binding(3) var<storage, read_write> atomicCount: atomic<u32>;

    @group(1) @binding(1) var<storage, read_write> outIndices: array<u32>;

    @compute @workgroup_size(${workgroupSize})
    fn main(
      @builtin(global_invocation_id) global_invocation_id: vec3u,
    ) {
      let cell_id: u32 = global_invocation_id.x;

      if (cell_id >= inNumCells) {
        return;
      }

      let data: vec4u = unpack4xU8(inNonEmptyCells[cell_id]);
      let coords: vec3u = data.xyz;
      let cube_case: u32 = data.w;

      if (any(coords >= vec3u(${kChunkDim} - 1))) {
        return;
      }

      let numTriangles: u32 = kNumTriangleLUT[cube_case];
      let offset: u32 = atomicAdd(&atomicCount, 3u * numTriangles);

      for (var i: u32 = 0u; i < numTriangles; i += 1u) {
        let tri_edge: vec3i = kTriangleLUT[5u * cube_case + i].xzy;

        for (var j: u32 = 0u; j < 3u; j += 1u) {
          let edge_id: u32 = u32(tri_edge[j]);

          var edge_coords: vec3u = coords + vec3u(kEdgeStart[edge_id]);
          edge_coords.x = 3u * edge_coords.x + kEdgeAxis[edge_id];

          let vertex_id: u32 = textureLoad(inIndicesVolume, edge_coords, 0).r;

          outIndices[offset + 3u * i + j] = vertex_id;
        }
      }
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    label,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: 'uint',
          viewDimension: vertexIndicesVolume.dimension,
          multisampled: false,
        }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
        }
      },
    ]
  });

  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({
      label,
      bindGroupLayouts: [bindGroupLayout, chunkBindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        label,
        code: computeShaderCode
      }),
    },
  });

  const bindGroup = device.createBindGroup({
    label,
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cellCountBuffer } },
      { binding: 1, resource: { buffer: nonEmptyCellsBuffer } },
      { binding: 2, resource: vertexIndicesVolume.createView({dimension: vertexIndicesVolume.dimension}) },
      { binding: 3, resource: { buffer: atomicCountBuffer } },
    ]
  });

  return {
    label,
    pipeline,
    bindGroup,
    indirect: {
      buffer: indirectBuffer
    }
  }
}

// ----------------------------------------------------------------------------
