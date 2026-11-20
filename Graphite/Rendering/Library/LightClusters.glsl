// The view's local lights and decals binned into clusters: a grid of tiles across the screen by slices of depth, each
// holding the items of the light and decal trees that may touch its box, found once a frame by
// Graphite/Internal/LightClusters.glsl. A fragment reads its cluster's short lists instead of walking the trees itself,
// so it keeps no traversal stack or hit list in its registers. See Graphite_BeginLights in Graphite/Lights.glsl and
// Graphite_ApplyDecals in Graphite/Decals.glsl.
//
// Each layer of the view has a block: a header of four uvec2, then the clusters, each GRAPHITE_CLUSTER_SLOTS uvec2. A
// cluster's lights come first: a count, then up to GRAPHITE_CLUSTER_MAX_LIGHTS hits. Its decals follow at
// GRAPHITE_CLUSTER_DECALS: a count, then up to GRAPHITE_CLUSTER_MAX_DECALS hits sorted lowest priority first, then the
// order they were added, the order they apply in. A hit is a record's index with the moving tree in the top bit and the
// region entry it was found through. The header holds the axis depth runs along, the near plane and the slice scale and
// bias, so slice = log2(depth) * scale + bias.
//
// A cluster holding more lights than fit keeps the ones lighting it most, by range over distance; one holding more
// decals keeps the highest priority ones, as a fragment does past GRAPHITE_MAX_DECALS.
#pragma once

#define GRAPHITE_CLUSTERS_X 16u
#define GRAPHITE_CLUSTERS_Y 9u
#define GRAPHITE_CLUSTERS_Z 24u
#define GRAPHITE_CLUSTER_COUNT (GRAPHITE_CLUSTERS_X * GRAPHITE_CLUSTERS_Y * GRAPHITE_CLUSTERS_Z)
#define GRAPHITE_CLUSTER_LIGHT_SLOTS 64u
#define GRAPHITE_CLUSTER_DECAL_SLOTS 32u
#define GRAPHITE_CLUSTER_SLOTS (GRAPHITE_CLUSTER_LIGHT_SLOTS + GRAPHITE_CLUSTER_DECAL_SLOTS)
#define GRAPHITE_CLUSTER_MAX_LIGHTS (GRAPHITE_CLUSTER_LIGHT_SLOTS - 1u)
#define GRAPHITE_CLUSTER_MAX_DECALS (GRAPHITE_CLUSTER_DECAL_SLOTS - 1u)
#define GRAPHITE_CLUSTER_DECALS GRAPHITE_CLUSTER_LIGHT_SLOTS
#define GRAPHITE_CLUSTER_HEADER 4u
#define GRAPHITE_CLUSTER_LAYER (GRAPHITE_CLUSTER_HEADER + GRAPHITE_CLUSTER_COUNT * GRAPHITE_CLUSTER_SLOTS)

// Slices run logarithmically from the near plane to this distance, the last reaching on to the view's far distance
#define GRAPHITE_CLUSTER_FAR 500.0

// The 16 byte words of a light's and a decal's record in their trees
#define GRAPHITE_LIGHT_WORDS 9u
#define GRAPHITE_DECAL_WORDS 10u

// The first uvec2 of a layer's block, and of a cluster in it
uint Graphite_ClusterLayerBase(uint layer) { return layer * GRAPHITE_CLUSTER_LAYER; }
uint Graphite_ClusterBase(uint layer, uint cluster) { return Graphite_ClusterLayerBase(layer) + GRAPHITE_CLUSTER_HEADER + cluster * GRAPHITE_CLUSTER_SLOTS; }

uint Graphite_ClusterIndex(uvec3 c) { return (c.z * GRAPHITE_CLUSTERS_Y + c.y) * GRAPHITE_CLUSTERS_X + c.x; }

#if defined(GRAPHITE_FRAGMENT) && !defined(GRAPHITE_NO_LIGHT_CLUSTERS)
#define GRAPHITE_CLUSTERED 1
layout(set = 1, binding = 50, std430) readonly buffer Graphite_LightClusters { uvec2 graphite_LightClusters[]; };

// The first uvec2 of the cluster a fragment at a camera relative position lies in, in the view layer it draws. A view
// without clusters binds empty ones for its first layer alone, whose header puts every fragment in its first slice, so
// a layer past the buffer reads those.
uint Graphite_FragmentCluster(vec3 relativePos)
{
    uint layer = graphite_Push.viewBase + GRAPHITE_VIEW_INDEX;
    if (Graphite_ClusterLayerBase(layer + 1u) > uint(graphite_LightClusters.length()))
        layer = 0u;
    uint layerBase = Graphite_ClusterLayerBase(layer);
    uvec2 h0 = graphite_LightClusters[layerBase], h1 = graphite_LightClusters[layerBase + 1u], h2 = graphite_LightClusters[layerBase + 2u];
    vec3 forward = vec3(uintBitsToFloat(h0), uintBitsToFloat(h1.x));
    vec2 slicing = uintBitsToFloat(h2);
    float depth = max(dot(relativePos, forward), uintBitsToFloat(h1.y));
    uvec2 tile = uvec2(clamp(gl_FragCoord.xy * Graphite_View().screen.zw, vec2(0.0), vec2(0.99999)) * vec2(GRAPHITE_CLUSTERS_X, GRAPHITE_CLUSTERS_Y));
    uint slice = uint(clamp(log2(depth) * slicing.x + slicing.y, 0.0, float(GRAPHITE_CLUSTERS_Z - 1u)));
    return Graphite_ClusterBase(layer, Graphite_ClusterIndex(uvec3(tile, slice)));
}
#endif
