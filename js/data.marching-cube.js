// ----------------------------------------------------------------------------

export const maxTrianglesPerVoxel = 5;

export const Shader_EdgesConstants = `
    const kEdgeStart: array<vec3f, 12> = array<vec3f, 12>(
      vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0),
      vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), vec3f(1.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0),
      vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
    );

    const kEdgeEnd: array<vec3f, 12> = array<vec3f, 12>(
      vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
      vec3f(0.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0), vec3f(1.0, 0.0, 1.0),
      vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 1.0), vec3f(1.0, 1.0, 1.0), vec3f(1.0, 0.0, 1.0),
    );

    const kEdgeDir: array<vec3f, 12> = array<vec3f, 12>(
      vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
      vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0),
      vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0),
    );

    const kEdgeAxis: array<u32, 12> = array<u32, 12>(
      1, 0, 1, 0,
      1, 0, 1, 0,
      2, 2, 2, 2,
    );
`;

export const Shader_kNumTriangleLUT = `
    const kNumTriangleLUT: array<u32, 256> = array<u32, 256>(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2,
      3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3,
      3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
      4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4,
      3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5,
      2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4,
      4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
      4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5,
      3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3,
      4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2,
      4, 1, 2, 1, 1, 0
    );
`;

export const Shader_kTriangleLUT = `
    const kTriangleLUT: array<vec3i, 1280> = array<vec3i, 1280>(
      vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 8, 3), vec3i(9, 8, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(1, 2, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 2, 10), vec3i(0, 2, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 8, 3), vec3i(2, 10, 8), vec3i(10, 9, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 11, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 11, 2), vec3i(8, 11, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 9, 0), vec3i(2, 3, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 11, 2), vec3i(1, 9, 11), vec3i(9, 8, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 10, 1), vec3i(11, 10, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 10, 1), vec3i(0, 8, 10), vec3i(8, 11, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 9, 0), vec3i(3, 11, 9), vec3i(11, 10, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 8, 10), vec3i(10, 8, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 7, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 3, 0), vec3i(7, 3, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(8, 4, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 1, 9), vec3i(4, 7, 1), vec3i(7, 3, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(8, 4, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 4, 7), vec3i(3, 0, 4), vec3i(1, 2, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 2, 10), vec3i(9, 0, 2), vec3i(8, 4, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 10, 9), vec3i(2, 9, 7), vec3i(2, 7, 3), vec3i(7, 9, 4), vec3i(-1, -1, -1),
      vec3i(8, 4, 7), vec3i(3, 11, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 4, 7), vec3i(11, 2, 4), vec3i(2, 0, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 0, 1), vec3i(8, 4, 7), vec3i(2, 3, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 7, 11), vec3i(9, 4, 11), vec3i(9, 11, 2), vec3i(9, 2, 1), vec3i(-1, -1, -1),
      vec3i(3, 10, 1), vec3i(3, 11, 10), vec3i(7, 8, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 11, 10), vec3i(1, 4, 11), vec3i(1, 0, 4), vec3i(7, 11, 4), vec3i(-1, -1, -1),
      vec3i(4, 7, 8), vec3i(9, 0, 11), vec3i(9, 11, 10), vec3i(11, 0, 3), vec3i(-1, -1, -1),
      vec3i(4, 7, 11), vec3i(4, 11, 9), vec3i(9, 11, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 5, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 5, 4), vec3i(0, 8, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 5, 4), vec3i(1, 5, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 5, 4), vec3i(8, 3, 5), vec3i(3, 1, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(9, 5, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 0, 8), vec3i(1, 2, 10), vec3i(4, 9, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 2, 10), vec3i(5, 4, 2), vec3i(4, 0, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 10, 5), vec3i(3, 2, 5), vec3i(3, 5, 4), vec3i(3, 4, 8), vec3i(-1, -1, -1),
      vec3i(9, 5, 4), vec3i(2, 3, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 11, 2), vec3i(0, 8, 11), vec3i(4, 9, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 5, 4), vec3i(0, 1, 5), vec3i(2, 3, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 1, 5), vec3i(2, 5, 8), vec3i(2, 8, 11), vec3i(4, 8, 5), vec3i(-1, -1, -1),
      vec3i(10, 3, 11), vec3i(10, 1, 3), vec3i(9, 5, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 9, 5), vec3i(0, 8, 1), vec3i(8, 10, 1), vec3i(8, 11, 10), vec3i(-1, -1, -1),
      vec3i(5, 4, 0), vec3i(5, 0, 11), vec3i(5, 11, 10), vec3i(11, 0, 3), vec3i(-1, -1, -1),
      vec3i(5, 4, 8), vec3i(5, 8, 10), vec3i(10, 8, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 7, 8), vec3i(5, 7, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 3, 0), vec3i(9, 5, 3), vec3i(5, 7, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 7, 8), vec3i(0, 1, 7), vec3i(1, 5, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 5, 3), vec3i(3, 5, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 7, 8), vec3i(9, 5, 7), vec3i(10, 1, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 1, 2), vec3i(9, 5, 0), vec3i(5, 3, 0), vec3i(5, 7, 3), vec3i(-1, -1, -1),
      vec3i(8, 0, 2), vec3i(8, 2, 5), vec3i(8, 5, 7), vec3i(10, 5, 2), vec3i(-1, -1, -1),
      vec3i(2, 10, 5), vec3i(2, 5, 3), vec3i(3, 5, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 9, 5), vec3i(7, 8, 9), vec3i(3, 11, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 5, 7), vec3i(9, 7, 2), vec3i(9, 2, 0), vec3i(2, 7, 11), vec3i(-1, -1, -1),
      vec3i(2, 3, 11), vec3i(0, 1, 8), vec3i(1, 7, 8), vec3i(1, 5, 7), vec3i(-1, -1, -1),
      vec3i(11, 2, 1), vec3i(11, 1, 7), vec3i(7, 1, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 5, 8), vec3i(8, 5, 7), vec3i(10, 1, 3), vec3i(10, 3, 11), vec3i(-1, -1, -1),
      vec3i(5, 7, 0), vec3i(5, 0, 9), vec3i(7, 11, 0), vec3i(1, 0, 10), vec3i(11, 10, 0),
      vec3i(11, 10, 0), vec3i(11, 0, 3), vec3i(10, 5, 0), vec3i(8, 0, 7), vec3i(5, 7, 0),
      vec3i(11, 10, 5), vec3i(7, 11, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 6, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(5, 10, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 0, 1), vec3i(5, 10, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 8, 3), vec3i(1, 9, 8), vec3i(5, 10, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 6, 5), vec3i(2, 6, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 6, 5), vec3i(1, 2, 6), vec3i(3, 0, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 6, 5), vec3i(9, 0, 6), vec3i(0, 2, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 9, 8), vec3i(5, 8, 2), vec3i(5, 2, 6), vec3i(3, 2, 8), vec3i(-1, -1, -1),
      vec3i(2, 3, 11), vec3i(10, 6, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 0, 8), vec3i(11, 2, 0), vec3i(10, 6, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(2, 3, 11), vec3i(5, 10, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 10, 6), vec3i(1, 9, 2), vec3i(9, 11, 2), vec3i(9, 8, 11), vec3i(-1, -1, -1),
      vec3i(6, 3, 11), vec3i(6, 5, 3), vec3i(5, 1, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 11), vec3i(0, 11, 5), vec3i(0, 5, 1), vec3i(5, 11, 6), vec3i(-1, -1, -1),
      vec3i(3, 11, 6), vec3i(0, 3, 6), vec3i(0, 6, 5), vec3i(0, 5, 9), vec3i(-1, -1, -1),
      vec3i(6, 5, 9), vec3i(6, 9, 11), vec3i(11, 9, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 10, 6), vec3i(4, 7, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 3, 0), vec3i(4, 7, 3), vec3i(6, 5, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 9, 0), vec3i(5, 10, 6), vec3i(8, 4, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 6, 5), vec3i(1, 9, 7), vec3i(1, 7, 3), vec3i(7, 9, 4), vec3i(-1, -1, -1),
      vec3i(6, 1, 2), vec3i(6, 5, 1), vec3i(4, 7, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 5), vec3i(5, 2, 6), vec3i(3, 0, 4), vec3i(3, 4, 7), vec3i(-1, -1, -1),
      vec3i(8, 4, 7), vec3i(9, 0, 5), vec3i(0, 6, 5), vec3i(0, 2, 6), vec3i(-1, -1, -1),
      vec3i(7, 3, 9), vec3i(7, 9, 4), vec3i(3, 2, 9), vec3i(5, 9, 6), vec3i(2, 6, 9),
      vec3i(3, 11, 2), vec3i(7, 8, 4), vec3i(10, 6, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 10, 6), vec3i(4, 7, 2), vec3i(4, 2, 0), vec3i(2, 7, 11), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(4, 7, 8), vec3i(2, 3, 11), vec3i(5, 10, 6), vec3i(-1, -1, -1),
      vec3i(9, 2, 1), vec3i(9, 11, 2), vec3i(9, 4, 11), vec3i(7, 11, 4), vec3i(5, 10, 6),
      vec3i(8, 4, 7), vec3i(3, 11, 5), vec3i(3, 5, 1), vec3i(5, 11, 6), vec3i(-1, -1, -1),
      vec3i(5, 1, 11), vec3i(5, 11, 6), vec3i(1, 0, 11), vec3i(7, 11, 4), vec3i(0, 4, 11),
      vec3i(0, 5, 9), vec3i(0, 6, 5), vec3i(0, 3, 6), vec3i(11, 6, 3), vec3i(8, 4, 7),
      vec3i(6, 5, 9), vec3i(6, 9, 11), vec3i(4, 7, 9), vec3i(7, 11, 9), vec3i(-1, -1, -1),
      vec3i(10, 4, 9), vec3i(6, 4, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 10, 6), vec3i(4, 9, 10), vec3i(0, 8, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 0, 1), vec3i(10, 6, 0), vec3i(6, 4, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 3, 1), vec3i(8, 1, 6), vec3i(8, 6, 4), vec3i(6, 1, 10), vec3i(-1, -1, -1),
      vec3i(1, 4, 9), vec3i(1, 2, 4), vec3i(2, 6, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 0, 8), vec3i(1, 2, 9), vec3i(2, 4, 9), vec3i(2, 6, 4), vec3i(-1, -1, -1),
      vec3i(0, 2, 4), vec3i(4, 2, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 3, 2), vec3i(8, 2, 4), vec3i(4, 2, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 4, 9), vec3i(10, 6, 4), vec3i(11, 2, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 2), vec3i(2, 8, 11), vec3i(4, 9, 10), vec3i(4, 10, 6), vec3i(-1, -1, -1),
      vec3i(3, 11, 2), vec3i(0, 1, 6), vec3i(0, 6, 4), vec3i(6, 1, 10), vec3i(-1, -1, -1),
      vec3i(6, 4, 1), vec3i(6, 1, 10), vec3i(4, 8, 1), vec3i(2, 1, 11), vec3i(8, 11, 1),
      vec3i(9, 6, 4), vec3i(9, 3, 6), vec3i(9, 1, 3), vec3i(11, 6, 3), vec3i(-1, -1, -1),
      vec3i(8, 11, 1), vec3i(8, 1, 0), vec3i(11, 6, 1), vec3i(9, 1, 4), vec3i(6, 4, 1),
      vec3i(3, 11, 6), vec3i(3, 6, 0), vec3i(0, 6, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(6, 4, 8), vec3i(11, 6, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 10, 6), vec3i(7, 8, 10), vec3i(8, 9, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 7, 3), vec3i(0, 10, 7), vec3i(0, 9, 10), vec3i(6, 7, 10), vec3i(-1, -1, -1),
      vec3i(10, 6, 7), vec3i(1, 10, 7), vec3i(1, 7, 8), vec3i(1, 8, 0), vec3i(-1, -1, -1),
      vec3i(10, 6, 7), vec3i(10, 7, 1), vec3i(1, 7, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 6), vec3i(1, 6, 8), vec3i(1, 8, 9), vec3i(8, 6, 7), vec3i(-1, -1, -1),
      vec3i(2, 6, 9), vec3i(2, 9, 1), vec3i(6, 7, 9), vec3i(0, 9, 3), vec3i(7, 3, 9),
      vec3i(7, 8, 0), vec3i(7, 0, 6), vec3i(6, 0, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 3, 2), vec3i(6, 7, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 3, 11), vec3i(10, 6, 8), vec3i(10, 8, 9), vec3i(8, 6, 7), vec3i(-1, -1, -1),
      vec3i(2, 0, 7), vec3i(2, 7, 11), vec3i(0, 9, 7), vec3i(6, 7, 10), vec3i(9, 10, 7),
      vec3i(1, 8, 0), vec3i(1, 7, 8), vec3i(1, 10, 7), vec3i(6, 7, 10), vec3i(2, 3, 11),
      vec3i(11, 2, 1), vec3i(11, 1, 7), vec3i(10, 6, 1), vec3i(6, 7, 1), vec3i(-1, -1, -1),
      vec3i(8, 9, 6), vec3i(8, 6, 7), vec3i(9, 1, 6), vec3i(11, 6, 3), vec3i(1, 3, 6),
      vec3i(0, 9, 1), vec3i(11, 6, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 8, 0), vec3i(7, 0, 6), vec3i(3, 11, 0), vec3i(11, 6, 0), vec3i(-1, -1, -1),
      vec3i(7, 11, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 6, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 0, 8), vec3i(11, 7, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(11, 7, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 1, 9), vec3i(8, 3, 1), vec3i(11, 7, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 1, 2), vec3i(6, 11, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(3, 0, 8), vec3i(6, 11, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 9, 0), vec3i(2, 10, 9), vec3i(6, 11, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(6, 11, 7), vec3i(2, 10, 3), vec3i(10, 8, 3), vec3i(10, 9, 8), vec3i(-1, -1, -1),
      vec3i(7, 2, 3), vec3i(6, 2, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(7, 0, 8), vec3i(7, 6, 0), vec3i(6, 2, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 7, 6), vec3i(2, 3, 7), vec3i(0, 1, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 6, 2), vec3i(1, 8, 6), vec3i(1, 9, 8), vec3i(8, 7, 6), vec3i(-1, -1, -1),
      vec3i(10, 7, 6), vec3i(10, 1, 7), vec3i(1, 3, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 7, 6), vec3i(1, 7, 10), vec3i(1, 8, 7), vec3i(1, 0, 8), vec3i(-1, -1, -1),
      vec3i(0, 3, 7), vec3i(0, 7, 10), vec3i(0, 10, 9), vec3i(6, 10, 7), vec3i(-1, -1, -1),
      vec3i(7, 6, 10), vec3i(7, 10, 8), vec3i(8, 10, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(6, 8, 4), vec3i(11, 8, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 6, 11), vec3i(3, 0, 6), vec3i(0, 4, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 6, 11), vec3i(8, 4, 6), vec3i(9, 0, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 4, 6), vec3i(9, 6, 3), vec3i(9, 3, 1), vec3i(11, 3, 6), vec3i(-1, -1, -1),
      vec3i(6, 8, 4), vec3i(6, 11, 8), vec3i(2, 10, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(3, 0, 11), vec3i(0, 6, 11), vec3i(0, 4, 6), vec3i(-1, -1, -1),
      vec3i(4, 11, 8), vec3i(4, 6, 11), vec3i(0, 2, 9), vec3i(2, 10, 9), vec3i(-1, -1, -1),
      vec3i(10, 9, 3), vec3i(10, 3, 2), vec3i(9, 4, 3), vec3i(11, 3, 6), vec3i(4, 6, 3),
      vec3i(8, 2, 3), vec3i(8, 4, 2), vec3i(4, 6, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 4, 2), vec3i(4, 6, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 9, 0), vec3i(2, 3, 4), vec3i(2, 4, 6), vec3i(4, 3, 8), vec3i(-1, -1, -1),
      vec3i(1, 9, 4), vec3i(1, 4, 2), vec3i(2, 4, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 1, 3), vec3i(8, 6, 1), vec3i(8, 4, 6), vec3i(6, 10, 1), vec3i(-1, -1, -1),
      vec3i(10, 1, 0), vec3i(10, 0, 6), vec3i(6, 0, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 6, 3), vec3i(4, 3, 8), vec3i(6, 10, 3), vec3i(0, 3, 9), vec3i(10, 9, 3),
      vec3i(10, 9, 4), vec3i(6, 10, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 9, 5), vec3i(7, 6, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(4, 9, 5), vec3i(11, 7, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 0, 1), vec3i(5, 4, 0), vec3i(7, 6, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 7, 6), vec3i(8, 3, 4), vec3i(3, 5, 4), vec3i(3, 1, 5), vec3i(-1, -1, -1),
      vec3i(9, 5, 4), vec3i(10, 1, 2), vec3i(7, 6, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(6, 11, 7), vec3i(1, 2, 10), vec3i(0, 8, 3), vec3i(4, 9, 5), vec3i(-1, -1, -1),
      vec3i(7, 6, 11), vec3i(5, 4, 10), vec3i(4, 2, 10), vec3i(4, 0, 2), vec3i(-1, -1, -1),
      vec3i(3, 4, 8), vec3i(3, 5, 4), vec3i(3, 2, 5), vec3i(10, 5, 2), vec3i(11, 7, 6),
      vec3i(7, 2, 3), vec3i(7, 6, 2), vec3i(5, 4, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 5, 4), vec3i(0, 8, 6), vec3i(0, 6, 2), vec3i(6, 8, 7), vec3i(-1, -1, -1),
      vec3i(3, 6, 2), vec3i(3, 7, 6), vec3i(1, 5, 0), vec3i(5, 4, 0), vec3i(-1, -1, -1),
      vec3i(6, 2, 8), vec3i(6, 8, 7), vec3i(2, 1, 8), vec3i(4, 8, 5), vec3i(1, 5, 8),
      vec3i(9, 5, 4), vec3i(10, 1, 6), vec3i(1, 7, 6), vec3i(1, 3, 7), vec3i(-1, -1, -1),
      vec3i(1, 6, 10), vec3i(1, 7, 6), vec3i(1, 0, 7), vec3i(8, 7, 0), vec3i(9, 5, 4),
      vec3i(4, 0, 10), vec3i(4, 10, 5), vec3i(0, 3, 10), vec3i(6, 10, 7), vec3i(3, 7, 10),
      vec3i(7, 6, 10), vec3i(7, 10, 8), vec3i(5, 4, 10), vec3i(4, 8, 10), vec3i(-1, -1, -1),
      vec3i(6, 9, 5), vec3i(6, 11, 9), vec3i(11, 8, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 6, 11), vec3i(0, 6, 3), vec3i(0, 5, 6), vec3i(0, 9, 5), vec3i(-1, -1, -1),
      vec3i(0, 11, 8), vec3i(0, 5, 11), vec3i(0, 1, 5), vec3i(5, 6, 11), vec3i(-1, -1, -1),
      vec3i(6, 11, 3), vec3i(6, 3, 5), vec3i(5, 3, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 10), vec3i(9, 5, 11), vec3i(9, 11, 8), vec3i(11, 5, 6), vec3i(-1, -1, -1),
      vec3i(0, 11, 3), vec3i(0, 6, 11), vec3i(0, 9, 6), vec3i(5, 6, 9), vec3i(1, 2, 10),
      vec3i(11, 8, 5), vec3i(11, 5, 6), vec3i(8, 0, 5), vec3i(10, 5, 2), vec3i(0, 2, 5),
      vec3i(6, 11, 3), vec3i(6, 3, 5), vec3i(2, 10, 3), vec3i(10, 5, 3), vec3i(-1, -1, -1),
      vec3i(5, 8, 9), vec3i(5, 2, 8), vec3i(5, 6, 2), vec3i(3, 8, 2), vec3i(-1, -1, -1),
      vec3i(9, 5, 6), vec3i(9, 6, 0), vec3i(0, 6, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 5, 8), vec3i(1, 8, 0), vec3i(5, 6, 8), vec3i(3, 8, 2), vec3i(6, 2, 8),
      vec3i(1, 5, 6), vec3i(2, 1, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 3, 6), vec3i(1, 6, 10), vec3i(3, 8, 6), vec3i(5, 6, 9), vec3i(8, 9, 6),
      vec3i(10, 1, 0), vec3i(10, 0, 6), vec3i(9, 5, 0), vec3i(5, 6, 0), vec3i(-1, -1, -1),
      vec3i(0, 3, 8), vec3i(5, 6, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 5, 6), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 5, 10), vec3i(7, 5, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 5, 10), vec3i(11, 7, 5), vec3i(8, 3, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 11, 7), vec3i(5, 10, 11), vec3i(1, 9, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(10, 7, 5), vec3i(10, 11, 7), vec3i(9, 8, 1), vec3i(8, 3, 1), vec3i(-1, -1, -1),
      vec3i(11, 1, 2), vec3i(11, 7, 1), vec3i(7, 5, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(1, 2, 7), vec3i(1, 7, 5), vec3i(7, 2, 11), vec3i(-1, -1, -1),
      vec3i(9, 7, 5), vec3i(9, 2, 7), vec3i(9, 0, 2), vec3i(2, 11, 7), vec3i(-1, -1, -1),
      vec3i(7, 5, 2), vec3i(7, 2, 11), vec3i(5, 9, 2), vec3i(3, 2, 8), vec3i(9, 8, 2),
      vec3i(2, 5, 10), vec3i(2, 3, 5), vec3i(3, 7, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 2, 0), vec3i(8, 5, 2), vec3i(8, 7, 5), vec3i(10, 2, 5), vec3i(-1, -1, -1),
      vec3i(9, 0, 1), vec3i(5, 10, 3), vec3i(5, 3, 7), vec3i(3, 10, 2), vec3i(-1, -1, -1),
      vec3i(9, 8, 2), vec3i(9, 2, 1), vec3i(8, 7, 2), vec3i(10, 2, 5), vec3i(7, 5, 2),
      vec3i(1, 3, 5), vec3i(3, 7, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 7), vec3i(0, 7, 1), vec3i(1, 7, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 0, 3), vec3i(9, 3, 5), vec3i(5, 3, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 8, 7), vec3i(5, 9, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 8, 4), vec3i(5, 10, 8), vec3i(10, 11, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(5, 0, 4), vec3i(5, 11, 0), vec3i(5, 10, 11), vec3i(11, 3, 0), vec3i(-1, -1, -1),
      vec3i(0, 1, 9), vec3i(8, 4, 10), vec3i(8, 10, 11), vec3i(10, 4, 5), vec3i(-1, -1, -1),
      vec3i(10, 11, 4), vec3i(10, 4, 5), vec3i(11, 3, 4), vec3i(9, 4, 1), vec3i(3, 1, 4),
      vec3i(2, 5, 1), vec3i(2, 8, 5), vec3i(2, 11, 8), vec3i(4, 5, 8), vec3i(-1, -1, -1),
      vec3i(0, 4, 11), vec3i(0, 11, 3), vec3i(4, 5, 11), vec3i(2, 11, 1), vec3i(5, 1, 11),
      vec3i(0, 2, 5), vec3i(0, 5, 9), vec3i(2, 11, 5), vec3i(4, 5, 8), vec3i(11, 8, 5),
      vec3i(9, 4, 5), vec3i(2, 11, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 5, 10), vec3i(3, 5, 2), vec3i(3, 4, 5), vec3i(3, 8, 4), vec3i(-1, -1, -1),
      vec3i(5, 10, 2), vec3i(5, 2, 4), vec3i(4, 2, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 10, 2), vec3i(3, 5, 10), vec3i(3, 8, 5), vec3i(4, 5, 8), vec3i(0, 1, 9),
      vec3i(5, 10, 2), vec3i(5, 2, 4), vec3i(1, 9, 2), vec3i(9, 4, 2), vec3i(-1, -1, -1),
      vec3i(8, 4, 5), vec3i(8, 5, 3), vec3i(3, 5, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 4, 5), vec3i(1, 0, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(8, 4, 5), vec3i(8, 5, 3), vec3i(9, 0, 5), vec3i(0, 3, 5), vec3i(-1, -1, -1),
      vec3i(9, 4, 5), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 11, 7), vec3i(4, 9, 11), vec3i(9, 10, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 8, 3), vec3i(4, 9, 7), vec3i(9, 11, 7), vec3i(9, 10, 11), vec3i(-1, -1, -1),
      vec3i(1, 10, 11), vec3i(1, 11, 4), vec3i(1, 4, 0), vec3i(7, 4, 11), vec3i(-1, -1, -1),
      vec3i(3, 1, 4), vec3i(3, 4, 8), vec3i(1, 10, 4), vec3i(7, 4, 11), vec3i(10, 11, 4),
      vec3i(4, 11, 7), vec3i(9, 11, 4), vec3i(9, 2, 11), vec3i(9, 1, 2), vec3i(-1, -1, -1),
      vec3i(9, 7, 4), vec3i(9, 11, 7), vec3i(9, 1, 11), vec3i(2, 11, 1), vec3i(0, 8, 3),
      vec3i(11, 7, 4), vec3i(11, 4, 2), vec3i(2, 4, 0), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(11, 7, 4), vec3i(11, 4, 2), vec3i(8, 3, 4), vec3i(3, 2, 4), vec3i(-1, -1, -1),
      vec3i(2, 9, 10), vec3i(2, 7, 9), vec3i(2, 3, 7), vec3i(7, 4, 9), vec3i(-1, -1, -1),
      vec3i(9, 10, 7), vec3i(9, 7, 4), vec3i(10, 2, 7), vec3i(8, 7, 0), vec3i(2, 0, 7),
      vec3i(3, 7, 10), vec3i(3, 10, 2), vec3i(7, 4, 10), vec3i(1, 10, 0), vec3i(4, 0, 10),
      vec3i(1, 10, 2), vec3i(8, 7, 4), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 9, 1), vec3i(4, 1, 7), vec3i(7, 1, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 9, 1), vec3i(4, 1, 7), vec3i(0, 8, 1), vec3i(8, 7, 1), vec3i(-1, -1, -1),
      vec3i(4, 0, 3), vec3i(7, 4, 3), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(4, 8, 7), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 10, 8), vec3i(10, 11, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 0, 9), vec3i(3, 9, 11), vec3i(11, 9, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 1, 10), vec3i(0, 10, 8), vec3i(8, 10, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 1, 10), vec3i(11, 3, 10), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 2, 11), vec3i(1, 11, 9), vec3i(9, 11, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 0, 9), vec3i(3, 9, 11), vec3i(1, 2, 9), vec3i(2, 11, 9), vec3i(-1, -1, -1),
      vec3i(0, 2, 11), vec3i(8, 0, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(3, 2, 11), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 3, 8), vec3i(2, 8, 10), vec3i(10, 8, 9), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(9, 10, 2), vec3i(0, 9, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(2, 3, 8), vec3i(2, 8, 10), vec3i(0, 1, 8), vec3i(1, 10, 8), vec3i(-1, -1, -1),
      vec3i(1, 10, 2), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(1, 3, 8), vec3i(9, 1, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 9, 1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(0, 3, 8), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
      vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1), vec3i(-1, -1, -1),
    );
`;

// ----------------------------------------------------------------------------
