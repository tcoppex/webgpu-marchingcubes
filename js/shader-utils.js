// ----------------------------------------------------------------------------

// Distance function helpers.
// ref: https://iquilezles.org/articles/distfunctions/

export const SignedDistanceFunctions = `
      fn opUnion(a: f32, b: f32) -> f32 {
        return min(a, b);
      }

      fn opIntersection(a: f32, b: f32) -> f32 {
        return max(a, b);
      }

      fn opSubstraction(a: f32, b: f32) -> f32 {
        return max(a, -b);
      }

      fn opSmoothUnion(a: f32, b: f32, k: f32) -> f32 {
        let res = exp(-k * a) + exp(-k * b);
        return - log(res) / k;
      }

      fn opRepeat(p: vec3f, c: vec3f) -> vec3f {
        return p - c * (floor(c / p) + 0.5);
      }

      fn opDisplacement(p: vec3f, d: vec3f) -> f32 {
        let dp = d*p;
        return sin(p.x) * sin(p.y) * sin(p.z);
      }

      fn sdPlane(p: vec3f, n: vec4f) -> f32 {
        return n.w + dot(p, n.xyz);
      }

      fn sdSphere(p: vec3f, r: f32) -> f32 {
        return length(p) - r;
      }

      fn udRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
        return length(max(abs(p) - b, vec3f(0.0))) - r;
      }

      fn sdCylinder(p: vec3f, r: f32) -> f32 {
        return length(p.xy) - r;
      }

      fn sdCylinder2(p: vec3f, c: vec3f) -> f32 {
        return opIntersection(length(p.xz - c.xy) - c.z, abs(p.y) - c.y);
      }

      fn sdTorus(p: vec3f, t: vec2f) -> f32 {
        let q = vec2f(length(p.xz) - t.x, p.y);
        return length(q) - t.y;
      }
`;

// ----------------------------------------------------------------------------

// Simplex Perlin Noise based on Stefan Gustavson & Ian McEwan implementation.

export const SimplexPerlinNoise = `
      override kPerlinNoisePermutationSeed: f32 = 0.0;
      override kNoiseTileRes: f32 = 512.0;
      override kNoiseEnableTiling: bool = false;


      fn mod289_vec3(x: vec3f) -> vec3f {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }

      fn mod289_vec4(x: vec4f) -> vec4f {
        return x - floor(x * (1.0 / 289.0)) * 289.0;
      }

      fn permute(x: vec4f) -> vec4f {
        return mod289_vec4(((x * 34.0) + 1.0) * x + kPerlinNoisePermutationSeed);
      }

      fn fade(u: vec3f) -> vec3f {
        return u * u * u * (u * (u * 6.0 - 15.0) + 10.0);
      }

      fn normalizeNoise(n: f32) -> f32 {
        return 0.5 * (2.44 * n + 1.0);
      }

      fn calculateGradients(p: vec4f) -> mat4x3f {
        var gx = p * (1.0 / 7.0);
        var gy = fract(floor(gx) * (1.0 / 7.0)) - 0.5;
        gx = fract(gx);
        let gz = vec4f(0.5) - abs(gx) - abs(gy);
        let sz = step(gz, vec4f(0.0));
        gx -= sz * (step(vec4f(0.0), gx) - 0.5);
        gy -= sz * (step(vec4f(0.0), gy) - 0.5);

        var g00 = vec3f(gx.x, gy.x, gz.x);
        var g10 = vec3f(gx.y, gy.y, gz.y);
        var g01 = vec3f(gx.z, gy.z, gz.z);
        var g11 = vec3f(gx.w, gy.w, gz.w);

        let dp = vec4f(dot(g00, g00), dot(g10, g10), dot(g01, g01), dot(g11, g11));
        let norm = inverseSqrt(dp);
        g00 *= norm.x;
        g10 *= norm.y;
        g01 *= norm.z;
        g11 *= norm.w;

        return mat4x3f(
          g00,
          g10,
          g01,
          g11,
        );
      }

      fn pnoise(pt: vec3f, scaledTileRes: vec3f) -> f32 {
        var ipt0 = floor(pt);
        var ipt1 = ipt0 + vec3f(1.0);

      	if (kNoiseEnableTiling) 
        {
          ipt0 = ipt0 % scaledTileRes;
          ipt1 = ipt1 % scaledTileRes;
      	}

        ipt0 = mod289_vec3(ipt0);
        ipt1 = mod289_vec3(ipt1);

        let ix = vec4f(ipt0.x, ipt1.x, ipt0.x, ipt1.x);
        let iy = vec4f(ipt0.yy, ipt1.yy);
        let p = permute(permute(ix) + iy);

        let fpt0 = fract(pt);
        let fpt1 = fpt0 - vec3f(1.0);

        let G0 = calculateGradients(permute(p + ipt0.zzzz));
        let n0 = vec4f(
          dot(G0[0], fpt0),
          dot(G0[1], vec3f(fpt1.x, fpt0.yz)),
          dot(G0[2], vec3f(fpt0.x, fpt1.y, fpt0.z)),
          dot(G0[3], vec3f(fpt1.xy, fpt0.z)),
        );

        let G1 = calculateGradients(permute(p + ipt1.zzzz));
        let n1 = vec4(
          dot(G1[0], vec3f(fpt0.xy, fpt1.z)),
          dot(G1[1], vec3f(fpt1.x, fpt0.y, fpt1.z)),
          dot(G1[2], vec3f(fpt0.x, fpt1.yz)),
          dot(G1[3], fpt1),
        );

        let u = fade(fpt0);
        return mix(
          mix(mix(n0.x, n0.y, u.x), mix(n0.z, n0.w, u.x), u.y), 
          mix(mix(n1.x, n1.y, u.x), mix(n1.z, n1.w, u.x), u.y), 
          u.z
        );
      }

      fn pnoise_3d(pt: vec3f) -> f32 {
        return pnoise(pt, vec3f(0.0));
      }

      fn pnoise_loop(u: vec2f, dt: f32) -> f32 {
        let pt1 = vec3f(u, dt);
        let pt2 = vec3f(u, dt - 1.0);
        return mix(pnoise_3d(pt1), pnoise_3d(pt2), dt);
      }

      fn fbm_pnoise(pt: vec3f, zoom: f32, numOctave: i32, frequency: f32, amplitude: f32) -> f32 {
        var sum = 0.0;
        var f = frequency;
        var w = amplitude;
        var v = zoom * pt;
        var scaledTileRes: vec3f = zoom * vec3f(kNoiseTileRes);
        for (var i = 0; i < numOctave; i = i + 1) {
          sum += w * pnoise(f * v, f * scaledTileRes);
          f *= frequency;
          w *= amplitude;
        }
        return sum;
      }

      fn fbm_3d(ws: vec3f) -> f32 {
        let N = 256.0;
        let zoom = 1.0 / N;
        let octave = 6;
        let freq = 2.5;
        let w = 0.45;
        return N * fbm_pnoise(ws, zoom, octave, freq, w);
      }
`;

// ----------------------------------------------------------------------------

// Hammersley point distribution on the sphere.

export const SphereRays64 = `
    const kSphereRays64 = array<vec3f, 64>(
      vec3f(-0.176084807, 0.000000000, 0.984375000),
      vec3f(0.000000000, 0.302576824, 0.953125000),
      vec3f(-0.000000000, -0.387487399, 0.921875000),
      vec3f(0.321548682, 0.321548682, 0.890625000),
      vec3f(-0.361576140, -0.361576140, 0.859375000),
      vec3f(-0.396364090, 0.396364090, 0.828125000),
      vec3f(0.427194472, -0.427194472, 0.796875000),
      vec3f(0.594319833, 0.246175335, 0.765625000),
      vec3f(-0.627077650, -0.259744067, 0.734375000),
      vec3f(-0.272113279, 0.656939569, 0.703125000),
      vec3f(0.283440054, -0.684284824, 0.671875000),
      vec3f(0.293844965, 0.709404501, 0.640625000),
      vec3f(-0.303422864, -0.732527593, 0.609375000),
      vec3f(-0.753837852, 0.312249862, 0.578125000),
      vec3f(0.773485129, -0.320388031, 0.546875000),
      vec3f(0.840350919, 0.167156191, 0.515625000),
      vec3f(-0.858050281, -0.170676813, 0.484375000),
      vec3f(-0.173912680, 0.874318085, 0.453125000),
      vec3f(0.176879422, -0.889232902, 0.421875000),
      vec3f(0.511430120, 0.765409264, 0.390625000),
      vec3f(-0.518454382, -0.775921815, 0.359375000),
      vec3f(-0.785434726, 0.524810705, 0.328125000),
      vec3f(0.793983927, -0.530523099, 0.296875000),
      vec3f(0.801600254, 0.535612166, 0.265625000),
      vec3f(-0.808310078, -0.540095527, 0.234375000),
      vec3f(-0.543988157, 0.814135811, 0.203125000),
      vec3f(0.547302663, -0.819096319, 0.171875000),
      vec3f(0.193151696, 0.971039150, 0.140625000),
      vec3f(-0.193919889, -0.974901116, 0.109375000),
      vec3f(-0.977787580, 0.194494042, 0.078125000),
      vec3f(0.979707165, -0.194875872, 0.046875000),
      vec3f(0.995063237, 0.098005175, 0.015625000),
      vec3f(-0.995063237, -0.098005175, -0.015625000),
      vec3f(-0.097909396, 0.994090783, -0.046875000),
      vec3f(0.097717558, -0.992143016, -0.078125000),
      vec3f(0.630587278, 0.768372822, -0.109375000),
      vec3f(-0.628089275, -0.765328996, -0.140625000),
      vec3f(-0.761507104, 0.624952730, -0.171875000),
      vec3f(0.756895361, -0.621167970, -0.203125000),
      vec3f(0.857356463, 0.458266577, -0.234375000),
      vec3f(-0.850239502, -0.454462482, -0.265625000),
      vec3f(-0.450144451, 0.842161034, -0.296875000),
      vec3f(0.445297532, -0.833093087, -0.328125000),
      vec3f(0.270891696, 0.893010245, -0.359375000),
      vec3f(-0.267221529, -0.880911326, -0.390625000),
      vec3f(-0.867613788, 0.263187765, -0.421875000),
      vec3f(0.853061581, -0.258773402, -0.453125000),
      vec3f(0.837189281, 0.253958592, -0.484375000),
      vec3f(-0.819920228, -0.248720082, -0.515625000),
      vec3f(-0.243030474, 0.801164105, -0.546875000),
      vec3f(0.236857263, -0.780813756, -0.578125000),
      vec3f(0.373762060, 0.699259631, -0.609375000),
      vec3f(-0.361963822, -0.677186681, -0.640625000),
      vec3f(-0.653207822, 0.349146855, -0.671875000),
      vec3f(0.627104460, -0.335194317, -0.703125000),
      vec3f(0.524676174, 0.430590608, -0.734375000),
      vec3f(-0.497267693, -0.408097049, -0.765625000),
      vec3f(-0.383265034, 0.467009795, -0.796875000),
      vec3f(0.355605014, -0.433305964, -0.828125000),
      vec3f(0.050120661, 0.508883610, -0.859375000),
      vec3f(-0.044572168, -0.452548816, -0.890625000),
      vec3f(-0.385621541, 0.037980407, -0.921875000),
      vec3f(0.301119834, -0.029657715, -0.953125000),
      vec3f(0.175872705, 0.008640072, -0.984375000),
    );
`;

export const SphereRays256 = `
    const kSphereRays256 = array<vec3f, 256>(
      vec3f(-0.088301989, 0.000000000, 0.996093750),
      vec3f(0.000000000, 0.152643935, 0.988281250),
      vec3f(-0.000000000, -0.196674936, 0.980468750),
      vec3f(0.164225180, 0.164225180, 0.972656250),
      vec3f(-0.185844744, -0.185844744, 0.964843750),
      vec3f(-0.205050221, 0.205050221, 0.957031250),
      vec3f(0.222467711, -0.222467711, 0.949218750),
      vec3f(0.311601260, 0.129069468, 0.941406250),
      vec3f(-0.331056745, -0.137128194, 0.933593750),
      vec3f(-0.144677153, 0.349281546, 0.925781250),
      vec3f(0.151792421, -0.366459322, 0.917968750),
      vec3f(0.158532403, 0.382731078, 0.910156250),
      vec3f(-0.164943110, -0.398207894, 0.902343750),
      vec3f(-0.412979156, 0.171061567, 0.894531250),
      vec3f(0.427118070, -0.176918097, 0.886718750),
      vec3f(0.467829237, 0.093057021, 0.878906250),
      vec3f(-0.481680096, -0.095812128, 0.871093750),
      vec3f(-0.098466607, 0.495025061, 0.863281250),
      vec3f(0.101028389, -0.507904009, 0.855468750),
      vec3f(0.294755473, 0.441132739, 0.847656250),
      vec3f(-0.301579200, -0.451345169, 0.839843750),
      vec3f(-0.461240108, 0.308190787, 0.832031250),
      vec3f(0.470837574, -0.314603609, 0.824218750),
      vec3f(0.480155405, 0.320829584, 0.816406250),
      vec3f(-0.489209579, -0.326879390, 0.808593750),
      vec3f(-0.332762635, 0.498014476, 0.800781250),
      vec3f(0.338488003, -0.506583096, 0.792968750),
      vec3f(0.120818991, 0.607398086, 0.785156250),
      vec3f(-0.122726652, -0.616988544, 0.777343750),
      vec3f(-0.626338461, 0.124586466, 0.769531250),
      vec3f(0.635458455, -0.126400546, 0.761718750),
      vec3f(0.653818466, 0.064395498, 0.753906250),
      vec3f(-0.662634703, -0.065263822, 0.746093750),
      vec3f(-0.066111874, 0.671245124, 0.738281250),
      vec3f(0.066940426, -0.679657550, 0.730468750),
      vec3f(0.438497459, 0.534310699, 0.722656250),
      vec3f(-0.443621210, -0.540554008, 0.714843750),
      vec3f(-0.546659314, 0.448631705, 0.707031250),
      vec3f(0.552631190, -0.453532697, 0.699218750),
      vec3f(0.637158297, 0.340568205, 0.691406250),
      vec3f(-0.643681450, -0.344054903, 0.683593750),
      vec3f(-0.347467587, 0.650066133, 0.675781250),
      vec3f(0.350808419, -0.656316389, 0.667968750),
      vec3f(0.218041037, 0.718784969, 0.660156250),
      vec3f(-0.220013502, -0.725287316, 0.652343750),
      vec3f(-0.731655497, 0.221945268, 0.644531250),
      vec3f(0.737892984, -0.223837390, 0.636718750),
      vec3f(0.744003067, 0.225690863, 0.628906250),
      vec3f(-0.749988857, -0.227506633, 0.621093750),
      vec3f(-0.229285595, 0.755853309, 0.613281250),
      vec3f(0.231028599, -0.761599225, 0.605468750),
      vec3f(0.377943493, 0.707082543, 0.597656250),
      vec3f(-0.380661067, -0.712166766, 0.589843750),
      vec3f(-0.717148756, 0.383323996, 0.582031250),
      vec3f(0.722030629, -0.385933412, 0.574218750),
      vec3f(0.637058152, 0.522820114, 0.566406250),
      vec3f(-0.641166866, -0.526192049, 0.558593750),
      vec3f(-0.529496124, 0.645192893, 0.550781250),
      vec3f(0.532733601, -0.649137772, 0.542968750),
      vec3f(0.082800282, 0.840685376, 0.535156250),
      vec3f(-0.083280462, -0.845560722, 0.527343750),
      vec3f(-0.850337033, 0.083750888, 0.519531250),
      vec3f(0.855015969, -0.084211723, 0.511718750),
      vec3f(0.862717918, 0.042382614, 0.503906250),
      vec3f(-0.867223094, -0.042603939, 0.496093750),
      vec3f(-0.042820688, 0.871635136, 0.488281250),
      vec3f(0.043032932, -0.875955448, 0.480468750),
      vec3f(0.591809236, 0.652960870, 0.472656250),
      vec3f(-0.594593417, -0.656032740, 0.464843750),
      vec3f(-0.659039451, 0.597318540, 0.457031250),
      vec3f(0.661981889, -0.599985411, 0.449218750),
      vec3f(0.811156258, 0.383648338, 0.441406250),
      vec3f(-0.814592401, -0.385273513, 0.433593750),
      vec3f(-0.386863021, 0.817953134, 0.425781250),
      vec3f(0.388417301, -0.821239383, 0.417968750),
      vec3f(0.307248690, 0.858702564, 0.410156250),
      vec3f(-0.308418844, -0.861972925, 0.402343750),
      vec3f(-0.865168387, 0.309562199, 0.394531250),
      vec3f(0.868289777, -0.310679050, 0.386718750),
      vec3f(0.897700933, 0.224862378, 0.378906250),
      vec3f(-0.900766566, -0.225630279, 0.371093750),
      vec3f(-0.226379658, 0.903758255, 0.363281250),
      vec3f(0.227110699, -0.906676733, 0.355468750),
      vec3f(0.482034075, 0.804225268, 0.347656250),
      vec3f(-0.483504331, -0.806678243, 0.339843750),
      vec3f(-0.809068282, 0.484936866, 0.332031250),
      vec3f(0.811395943, -0.486332012, 0.324218750),
      vec3f(0.761941768, 0.565094529, 0.316406250),
      vec3f(-0.764006131, -0.566625564, 0.308593750),
      vec3f(-0.568114351, 0.766013527, 0.300781250),
      vec3f(0.569561221, -0.767964405, 0.292968750),
      vec3f(0.140638378, 0.948106935, 0.285156250),
      vec3f(-0.140974349, -0.950371863, 0.277343750),
      vec3f(-0.952568713, 0.141300221, 0.269531250),
      vec3f(0.954697956, -0.141616064, 0.261718750),
      vec3f(0.956760042, 0.141921946, 0.253906250),
      vec3f(-0.958755405, -0.142217930, 0.246093750),
      vec3f(-0.142504078, 0.960684461, 0.238281250),
      vec3f(0.142780450, -0.962547608, 0.230468750),
      vec3f(0.580745475, 0.783044627, 0.222656250),
      vec3f(-0.581788793, -0.784451378, 0.214843750),
      vec3f(-0.785805502, 0.582793080, 0.207031250),
      vec3f(0.787107272, -0.583758539, 0.199218750),
      vec3f(0.841869980, 0.504597447, 0.191406250),
      vec3f(-0.843149115, -0.505364131, 0.183593750),
      vec3f(-0.506097780, 0.844373134, 0.175781250),
      vec3f(0.506798537, -0.845542277, 0.167968750),
      vec3f(0.239843713, 0.957509775, 0.160156250),
      vec3f(-0.240144010, -0.958708629, 0.152343750),
      vec3f(-0.959846154, 0.240428945, 0.144531250),
      vec3f(0.960922567, -0.240698573, 0.136718750),
      vec3f(0.933688560, 0.334079109, 0.128906250),
      vec3f(-0.934615313, -0.334410706, 0.121093750),
      vec3f(-0.334721280, 0.935483310, 0.113281250),
      vec3f(0.335010890, -0.936292714, 0.105468750),
      vec3f(0.425511468, 0.899668410, 0.097656250),
      vec3f(-0.425826006, -0.900333445, 0.089843750),
      vec3f(-0.900942630, 0.426114129, 0.082031250),
      vec3f(0.901496077, -0.426375890, 0.074218750),
      vec3f(0.739315601, 0.670076602, 0.066406250),
      vec3f(-0.739678105, -0.670405156, 0.058593750),
      vec3f(-0.670692509, 0.739995150, 0.050781250),
      vec3f(0.670938714, -0.740266795, 0.042968750),
      vec3f(0.049037342, 0.998178029, 0.035156250),
      vec3f(-0.049049327, -0.998421996, 0.027343750),
      vec3f(-0.998604933, 0.049058315, 0.019531250),
      vec3f(0.998726872, -0.049064305, 0.011718750),
      vec3f(0.999691192, 0.024541041, 0.003906250),
      vec3f(-0.999691192, -0.024541041, -0.003906250),
      vec3f(-0.024539543, 0.999630172, -0.011718750),
      vec3f(0.024536547, -0.999508123, -0.019531250),
      vec3f(0.689282718, 0.723976280, -0.027343750),
      vec3f(-0.689114290, -0.723799374, -0.035156250),
      vec3f(-0.723578180, 0.688903697, -0.042968750),
      vec3f(0.723312659, -0.688650900, -0.050781250),
      vec3f(0.912639061, 0.404545073, -0.058593750),
      vec3f(-0.912191792, -0.404346812, -0.066406250),
      vec3f(-0.404123652, 0.911688352, -0.074218750),
      vec3f(0.403875553, -0.911128647, -0.082031250),
      vec3f(0.358439575, 0.929219657, -0.089843750),
      vec3f(-0.358174812, -0.928533285, -0.097656250),
      vec3f(-0.927789141, 0.357887764, -0.105468750),
      vec3f(0.926987088, -0.357578378, -0.113281250),
      vec3f(0.968522011, 0.217488891, -0.121093750),
      vec3f(-0.967561637, -0.217273231, -0.128906250),
      vec3f(-0.217043859, 0.966540194, -0.136718750),
      vec3f(0.216800729, -0.965457488, -0.144531250),
      vec3f(0.528752897, 0.834992070, -0.152343750),
      vec3f(-0.528091697, -0.833947921, -0.160156250),
      vec3f(-0.832850157, 0.527396545, -0.167968750),
      vec3f(0.831698563, -0.526667307, -0.175781250),
      vec3f(0.803687674, 0.566020722, -0.183593750),
      vec3f(-0.802468406, -0.565162016, -0.191406250),
      vec3f(-0.564266142, 0.801196362, -0.199218750),
      vec3f(0.563332921, -0.799871291, -0.207031250),
      vec3f(0.166969661, 0.962269866, -0.214843750),
      vec3f(-0.166670235, -0.960544235, -0.222656250),
      vec3f(-0.958753699, 0.166359548, -0.230468750),
      vec3f(0.956897896, -0.166037536, -0.238281250),
      vec3f(0.961956849, 0.118646061, -0.246093750),
      vec3f(-0.959954822, -0.118399135, -0.253906250),
      vec3f(-0.118143951, 0.957885851, -0.261718750),
      vec3f(0.117880457, -0.955749498, -0.269531250),
      vec3f(0.591096520, 0.757420193, -0.277343750),
      vec3f(-0.589687818, -0.755615108, -0.285156250),
      vec3f(-0.753755377, 0.588236469, -0.292968750),
      vec3f(0.751840594, -0.586742159, -0.300781250),
      vec3f(0.827621467, 0.468841770, -0.308593750),
      vec3f(-0.825385215, -0.467574949, -0.316406250),
      vec3f(-0.466272885, 0.823086751, -0.324218750),
      vec3f(0.464935283, -0.820725552, -0.332031250),
      vec3f(0.250838524, 0.906413956, -0.339843750),
      vec3f(-0.250075766, -0.903657702, -0.347656250),
      vec3f(-0.900830083, 0.249293258, -0.355468750),
      vec3f(0.897930425, -0.248490815, -0.363281250),
      vec3f(0.881727507, 0.291283423, -0.371093750),
      vec3f(-0.878726671, -0.290292081, -0.378906250),
      vec3f(-0.289276581, 0.875652710, -0.386718750),
      vec3f(0.288236669, -0.872504851, -0.394531250),
      vec3f(0.411614079, 0.817736728, -0.402343750),
      vec3f(-0.410052398, -0.814634201, -0.410156250),
      vec3f(-0.811459803, 0.408454540, -0.417968750),
      vec3f(0.808212688, -0.406820080, -0.425781250),
      vec3f(0.682327299, 0.588579576, -0.433593750),
      vec3f(-0.679449081, -0.586096808, -0.441406250),
      vec3f(-0.583558858, 0.676506891, -0.449218750),
      vec3f(0.580965002, -0.673499891, -0.457031250),
      vec3f(0.065133530, 0.882993721, -0.464843750),
      vec3f(-0.064828542, -0.878859107, -0.472656250),
      vec3f(-0.874635546, 0.064516994, -0.480468750),
      vec3f(0.870321743, -0.064198789, -0.488281250),
      vec3f(0.865916350, 0.063873827, -0.496093750),
      vec3f(-0.861417962, -0.063542006, -0.503906250),
      vec3f(-0.063203217, 0.856825113, -0.511718750),
      vec3f(0.062857348, -0.852136277, -0.519531250),
      vec3f(0.554969631, 0.643364032, -0.527343750),
      vec3f(-0.551769779, -0.639654515, -0.535156250),
      vec3f(-0.635868326, 0.548503790, -0.542968750),
      vec3f(0.632004088, -0.545170476, -0.550781250),
      vec3f(0.740877208, 0.372926247, -0.558593750),
      vec3f(-0.736129531, -0.370536467, -0.566406250),
      vec3f(-0.368097657, 0.731284447, -0.574218750),
      vec3f(0.365608835, -0.726340006, -0.582031250),
      vec3f(0.253303463, 0.766760527, -0.589843750),
      vec3f(-0.251495107, -0.761286554, -0.597656250),
      vec3f(-0.755700120, 0.249649598, -0.605468750),
      vec3f(0.749998710, -0.247766107, -0.613281250),
      vec3f(0.755346267, 0.209032464, -0.621093750),
      vec3f(-0.749317718, -0.207364140, -0.628906250),
      vec3f(-0.205661174, 0.743163989, -0.636718750),
      vec3f(0.203922698, -0.736881945, -0.644531250),
      vec3f(0.373578993, 0.659459148, -0.652343750),
      vec3f(-0.370229782, -0.653546964, -0.660156250),
      vec3f(-0.647509449, 0.366809572, -0.667968750),
      vec3f(0.641343064, -0.363316358, -0.675781250),
      vec3f(0.575384665, 0.449034600, -0.683593750),
      vec3f(-0.569553641, -0.444484024, -0.691406250),
      vec3f(-0.439833853, 0.563594998, -0.699218750),
      vec3f(0.435080894, -0.557504643, -0.707031250),
      vec3f(0.085599853, 0.694025272, -0.714843750),
      vec3f(-0.084611189, -0.686009396, -0.722656250),
      vec3f(-0.677810049, 0.083599896, -0.730468750),
      vec3f(0.669420491, -0.082565143, -0.738281250),
      vec3f(0.656038161, 0.113833419, -0.746093750),
      vec3f(-0.647309689, -0.112318886, -0.753906250),
      vec3f(-0.110767545, 0.638369091, -0.761718750),
      vec3f(0.109177828, -0.629207325, -0.769531250),
      vec3f(0.362227151, 0.514323036, -0.777343750),
      vec3f(-0.356596699, -0.506328409, -0.785156250),
      vec3f(-0.498123611, 0.350818228, -0.792968750),
      vec3f(0.489698080, -0.344884300, -0.800781250),
      vec3f(0.497084260, 0.314775136, -0.808593750),
      vec3f(-0.487884343, -0.308949353, -0.816406250),
      vec3f(-0.302953924, 0.478416526, -0.824218750),
      vec3f(0.296778567, -0.468664572, -0.832031250),
      vec3f(0.118934336, 0.529638650, -0.839843750),
      vec3f(-0.116243250, -0.517654700, -0.847656250),
      vec3f(-0.505271677, 0.113462549, -0.855468750),
      vec3f(0.492459477, -0.110585474, -0.863281250),
      vec3f(0.458208407, 0.176750487, -0.871093750),
      vec3f(-0.445032483, -0.171667972, -0.878906250),
      vec3f(-0.166382811, 0.431331217, -0.886718750),
      vec3f(0.160875031, -0.417052835, -0.894531250),
      vec3f(0.174665944, 0.394040055, -0.902343750),
      vec3f(-0.167877347, -0.378725226, -0.910156250),
      vec3f(-0.362623779, 0.160740066, -0.917968750),
      vec3f(0.345625794, -0.153205378, -0.925781250),
      vec3f(0.259521803, 0.247085297, -0.933593750),
      vec3f(-0.244270270, -0.232564631, -0.941406250),
      vec3f(-0.216941077, 0.227860339, -0.949218750),
      vec3f(0.199956280, -0.210020648, -0.957031250),
      vec3f(0.006450028, 0.262745000, -0.964843750),
      vec3f(-0.005699687, -0.232179527, -0.972656250),
      vec3f(-0.196615701, 0.004826645, -0.980468750),
      vec3f(0.152597962, -0.003746070, -0.988281250),
      vec3f(0.088295340, 0.001083601, -0.996093750),
    );
`;

// ----------------------------------------------------------------------------
