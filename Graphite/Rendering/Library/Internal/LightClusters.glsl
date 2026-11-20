// Bins a view layer's local lights and decals into its clusters, see Graphite/LightClusters.glsl: a thread a cluster
// finds its box in camera relative space, walks the light and decal trees with it and keeps what may touch the box.
// Lights and decals on none of _ReceiverLayers, the layers of everything the view may draw, are left out. Lights whose
// sphere reaches the box are kept, the ones lighting it least giving way when it is full. Decals whose box overlaps it
// are kept sorted in the order they apply, the lowest priority giving way when it is full. The first slice starts at the
// camera and the last reaches past anything the view draws. The first thread writes the layer's header. Bound to the
// view's tables, one dispatch a layer with its view base pushed.
Compute
{
    #pragma resource_set 3
    #define GRAPHITE_NO_OBJECT 1

    layout(local_size_x = 64) in;
    #include "Graphite/Core.glsl"
    #include "Graphite/Trees.glsl"
    #include "Graphite/LightClusters.glsl"

    layout(std430) buffer Graphite_LightClusterWrite { uvec2 clusters[]; };

    uniform int _ReceiverLayers;

    // How much a light lights a box at d2 from its centre: its luminous intensity, times the area of a shape in nits, over
    // the distance squared, never more than at a tenth of its range
    float LightWeight(uint array, uint k, float d2, float range)
    {
        uint w = k * GRAPHITE_LIGHT_WORDS;
        vec4 colorIntensity = Graphite_SpatialVec4(array, w + 1u);
        uint type = floatBitsToUint(Graphite_SpatialVec4(array, w + 2u).w) & 15u;
        float size = Graphite_SpatialVec4(array, w + 3u).w;
        vec2 shape = Graphite_SpatialVec4(array, w + 4u).xy;
        float area = type == 3u ? size * shape.x : type == 4u ? 3.14159265 * size * size : type == 5u ? 2.0 * shape.y * size : 1.0;
        float luminance = dot(colorIntensity.rgb, vec3(0.2126, 0.7152, 0.0722)) * colorIntensity.w * area;
        return luminance / max(d2, 0.01 * range * range + 1e-6);
    }

    // A point on the ray of a screen uv at view depth z along forward
    vec3 CornerAt(Graphite_ViewData view, vec2 uv, vec3 forward, float z)
    {
        vec3 a = Graphite_RelativeFromDepth(view, uv, 1.0);
        vec3 b = Graphite_RelativeFromDepth(view, uv, 0.5);
        vec3 d = b - a;
        float along = dot(d, forward);
        float t = abs(along) > 1e-12 ? (z - dot(a, forward)) / along : 0.0;
        return a + d * t;
    }

    // Kept lights' importance, to find the one giving way
    float lightWeight[GRAPHITE_CLUSTER_MAX_LIGHTS];

    // Kept decals' order: priority, then the order they were added in
    float decalPriority[GRAPHITE_CLUSTER_MAX_DECALS];
    uint decalSequence[GRAPHITE_CLUSTER_MAX_DECALS];

    bool DecalBefore(float priority, uint sequence, float otherPriority, uint otherSequence)
    {
        return priority < otherPriority || (priority == otherPriority && sequence < otherSequence);
    }

    void main()
    {
        uint cluster = gl_GlobalInvocationID.x;
        if (cluster >= GRAPHITE_CLUSTER_COUNT)
            return;
        uint layer = graphite_Push.viewBase;
        Graphite_ViewData view = Graphite_View();

        // Depth runs along the view's z axis, pointing the way the near plane lies from the camera
        vec3 axis = vec3(view.view[0][2], view.view[1][2], view.view[2][2]);
        vec3 nearCenter = Graphite_RelativeFromDepth(view, vec2(0.5), 1.0);
        vec3 forward = dot(axis, nearCenter) < 0.0 ? -axis : axis;
        float near = max(view.depth.x, 1e-4);
        float sliceFar = max(min(GRAPHITE_CLUSTER_FAR, view.depth.y), near * 2.0);
        float scale = float(GRAPHITE_CLUSTERS_Z) / log2(sliceFar / near);
        float bias = -log2(near) * scale;
        uint layerBase = Graphite_ClusterLayerBase(layer);
        if (cluster == 0u)
        {
            clusters[layerBase] = floatBitsToUint(forward.xy);
            clusters[layerBase + 1u] = uvec2(floatBitsToUint(forward.z), floatBitsToUint(near));
            clusters[layerBase + 2u] = floatBitsToUint(vec2(scale, bias));
            clusters[layerBase + 3u] = uvec2(0u);
        }

        uvec3 c = uvec3(cluster % GRAPHITE_CLUSTERS_X, (cluster / GRAPHITE_CLUSTERS_X) % GRAPHITE_CLUSTERS_Y, cluster / (GRAPHITE_CLUSTERS_X * GRAPHITE_CLUSTERS_Y));
        // A fragment nearer than the first slice's far end, the near plane's other side included, falls in it, and one past
        // the last slice's near end however far
        float z0 = c.z == 0u ? min(view.depth.x, 0.0) : exp2((float(c.z) - bias) / scale);
        float z1 = c.z + 1u == GRAPHITE_CLUSTERS_Z ? 1e6 : exp2((float(c.z + 1u) - bias) / scale);
        vec2 uv0 = vec2(c.xy) / vec2(GRAPHITE_CLUSTERS_X, GRAPHITE_CLUSTERS_Y);
        vec2 uv1 = vec2(c.xy + 1u) / vec2(GRAPHITE_CLUSTERS_X, GRAPHITE_CLUSTERS_Y);

        vec3 lo = vec3(3.0e38), hi = vec3(-3.0e38);
        for (int i = 0; i < 4; i++)
        {
            vec2 uv = vec2((i & 1) != 0 ? uv1.x : uv0.x, (i & 2) != 0 ? uv1.y : uv0.y);
            vec3 p0 = CornerAt(view, uv, forward, z0), p1 = CornerAt(view, uv, forward, z1);
            lo = min(lo, min(p0, p1));
            hi = max(hi, max(p0, p1));
        }
        vec3 center = (lo + hi) * 0.5;
        vec3 extent = (hi - lo) * 0.5;
        uint base = Graphite_ClusterBase(layer, cluster);
        uint receivers = uint(_ReceiverLayers);

        // Lights whose sphere reaches the box, weighted by range over distance to it
        uint count = 0u;
        Graphite_TreeQuery q;
        Graphite_BeginBoxQuery(q, center, extent, GRAPHITE_SET_LIGHTS);
        uint array, first, last;
        while (Graphite_NextRun(q, array, first, last))
        {
            for (uint k = first; k < last; k++)
            {
                vec4 positionRange = Graphite_SpatialVec4(array, k * GRAPHITE_LIGHT_WORDS);
                precise vec3 position = (positionRange.xyz - q.offset) - q.offsetLow;
                vec3 d = max(lo - position, vec3(0.0)) + max(position - hi, vec3(0.0));
                float d2 = dot(d, d);
                if (d2 > positionRange.w * positionRange.w)
                    continue;
                if ((floatBitsToUint(Graphite_SpatialVec4(array, k * GRAPHITE_LIGHT_WORDS + 6u).w) & receivers) == 0u)
                    continue;
                float weight = LightWeight(array, k, d2, positionRange.w);
                uint slot = count;
                if (count == GRAPHITE_CLUSTER_MAX_LIGHTS)
                {
                    slot = 0u;
                    for (uint s = 1u; s < count; s++)
                        slot = lightWeight[s] < lightWeight[slot] ? s : slot;
                    if (lightWeight[slot] >= weight)
                        continue;
                }
                else
                {
                    count++;
                }
                lightWeight[slot] = weight;
                clusters[base + 1u + slot] = uvec2(k | ((array & 1u) << 31), q.region);
            }
        }
        clusters[base] = uvec2(count, 0u);

        // Decals whose box overlaps the cluster's: the cluster's box in the decal's unit box space, grown by its extent
        // there, holds the decal's half unit box
        uint decalBase = base + GRAPHITE_CLUSTER_DECALS;
        count = 0u;
        Graphite_BeginBoxQuery(q, center, extent, GRAPHITE_SET_DECALS);
        while (Graphite_NextRun(q, array, first, last))
        {
            for (uint k = first; k < last; k++)
            {
                uint w = k * GRAPHITE_DECAL_WORDS;
                vec4 row0 = Graphite_SpatialVec4(array, w), row1 = Graphite_SpatialVec4(array, w + 1u), row2 = Graphite_SpatialVec4(array, w + 2u);
                precise vec3 decalCenter = (vec3(row0.w, row1.w, row2.w) - q.offset) - q.offsetLow;
                vec3 toCluster = center - decalCenter;
                vec3 local = vec3(dot(row0.xyz, toCluster), dot(row1.xyz, toCluster), dot(row2.xyz, toCluster));
                vec3 reach = vec3(dot(abs(row0.xyz), extent), dot(abs(row1.xyz), extent), dot(abs(row2.xyz), extent));
                if (any(greaterThan(abs(local), reach + 0.5)))
                    continue;
                vec4 params = Graphite_SpatialVec4(array, w + 6u);
                if ((floatBitsToUint(params.w) & receivers) == 0u)
                    continue;

                float priority = params.y;
                uint sequence = Graphite_SpatialUVec4(array, w + 3u).x;
                uint at = count;
                if (count == GRAPHITE_CLUSTER_MAX_DECALS)
                {
                    if (DecalBefore(priority, sequence, decalPriority[0], decalSequence[0]))
                        continue;
                    for (uint s = 0u; s + 1u < count; s++)
                    {
                        decalPriority[s] = decalPriority[s + 1u];
                        decalSequence[s] = decalSequence[s + 1u];
                        clusters[decalBase + 1u + s] = clusters[decalBase + 2u + s];
                    }
                    at = count - 1u;
                }
                else
                {
                    count++;
                }
                while (at > 0u && DecalBefore(priority, sequence, decalPriority[at - 1u], decalSequence[at - 1u]))
                {
                    decalPriority[at] = decalPriority[at - 1u];
                    decalSequence[at] = decalSequence[at - 1u];
                    clusters[decalBase + 1u + at] = clusters[decalBase + at];
                    at--;
                }
                decalPriority[at] = priority;
                decalSequence[at] = sequence;
                clusters[decalBase + 1u + at] = uvec2(k | ((array & 1u) << 31), q.region);
            }
        }
        clusters[decalBase] = uvec2(count, 0u);
    }
}
